
/// <reference lib="webworker" />

import * as ort from 'onnxruntime-web/wasm';
import type { ModelLoadStage, WorkerMessage } from '../type';
import { getMaxSessionThreads } from '../config/concurrency';

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';

const isElectron =
  typeof navigator !== 'undefined' &&
  navigator.userAgent.includes('Electron');

ort.env.wasm.numThreads = isElectron ? 1 : getMaxSessionThreads();
ort.env.wasm.initTimeout = 30_000;

console.log(
  `[worker] init numThreads=${ort.env.wasm.numThreads} electron=${isElectron} isolated=${typeof self !== 'undefined' && self.crossOriginIsolated}`,
);

// ── Tiling ────────────────────────────────────────────────────────────────────

const DEFAULT_TILE_SIZE = 256;
const TILE_SIZE_GRANULARITY = 64;
const MAX_TILE_SIZE = 1024;
const TILE_OVERLAP = 16;

function normalizeTileSize(size?: number): number {
  if (size == null || !Number.isFinite(size)) return DEFAULT_TILE_SIZE;
  const snapped = Math.round(size / TILE_SIZE_GRANULARITY) * TILE_SIZE_GRANULARITY;
  return Math.min(MAX_TILE_SIZE, Math.max(TILE_SIZE_GRANULARITY, snapped));
}

const INFER_TIMEOUT_MS = 300_000;
const MAX_OUTPUT_DIMENSION = 16_384;
const MAX_OUTPUT_PIXELS = 200_000_000;

class AbortError extends Error {
  constructor() {
    super('Processing aborted');
    this.name = 'AbortError';
  }
}

const FATAL_PATTERN =
  /memory access out of bounds|cannot enlarge memory|out of memory|allocation failed|RangeError|Aborted|RuntimeError|Assertion failed|OOM/i;

let session: ort.InferenceSession | null = null;
let currentModelId: string | null = null;
let currentGpu = false;
let currentSource: string | ArrayBuffer | Uint8Array | null = null;

let sessionThreads = isElectron ? 1 : getMaxSessionThreads();
let currentTileSize = DEFAULT_TILE_SIZE;
let wantsGpu = true;
let gpuPreferredLayout: 'NCHW' | 'NHWC' | undefined;
let gpuInferTimeoutMs: number | null = null;
let sessionStale = false;

let abortRequested = false;

function yieldToEventLoop(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

let currentJobId: number | null = null;
let jobChain: Promise<void> = Promise.resolve();

let providedWasmBuffer: SharedArrayBuffer | null = null;
let wasmBinaryApplied = false;
const providedModelBuffers = new Map<string, Uint8Array>();

type GpuDeviceLike = { lost: Promise<{ reason?: string }> };
type GpuAdapterLike = { requestDevice(): Promise<GpuDeviceLike> };

let adapterProbe: Promise<GpuAdapterLike | null> | null = null;
let gpuDevice: GpuDeviceLike | null = null;
let deviceMonitorStarted = false;
let gpuFailed = false;
let gpuFailedModelId: string | null = null;

const workerScope = self as unknown as { name?: string };
const isOrtPthreadWorker = workerScope.name?.startsWith('em-pthread') ?? false;

if (!isOrtPthreadWorker) {
  self.onmessage = async (e: MessageEvent<WorkerMessage>) => {
    const message = e.data;

    switch (message.type) {
      case 'load-model':
        await loadModel(
          message.payload.modelId,
          message.payload.modelData,
          message.payload.gpu ?? false,
          message.payload.numThreads,
          message.payload.warmup ?? false,
          message.payload.tileSize,
          message.payload.preferredLayout,
          message.payload.gpuProbeTimeoutMs,
        );
        break;

      case 'process': {
        const payload = message.payload;
        jobChain = jobChain.then(() => processImage(payload));
        await jobChain;
        break;
      }

      case 'configure-worker':
        configureWorker(message.payload.numThreads, message.payload.gpu);
        break;

      case 'abort':
        abortRequested = true;
        break;

      case 'provide-wasm':
        providedWasmBuffer = message.payload.buffer;
        applyProvidedWasmBinary();
        break;

      case 'provide-model':
        providedModelBuffers.set(
          message.payload.modelId,
          new Uint8Array(message.payload.buffer),
        );
        console.log(
          `[worker] received model buffer: ${message.payload.modelId}, bytes=${message.payload.buffer.byteLength}`,
        );
        break;
    }
  };
}

function applyProvidedWasmBinary() {
  if (wasmBinaryApplied || !providedWasmBuffer) return;

  wasmBinaryApplied = true;

  const bytes = new Uint8Array(providedWasmBuffer.byteLength);
  bytes.set(new Uint8Array(providedWasmBuffer));

  (ort.env.wasm as unknown as { wasmBinary?: ArrayBuffer }).wasmBinary =
    bytes.buffer;
}

function resetAdapterProbe() {
  adapterProbe = null;
  gpuDevice = null;
}

async function probeWebGpuAdapter(): Promise<GpuAdapterLike | null> {
  if (adapterProbe) return adapterProbe;

  adapterProbe = (async () => {
    try {
      const gpu = (
        navigator as unknown as {
          gpu?: {
            requestAdapter(options?: {
              powerPreference?: 'low-power' | 'high-performance';
            }): Promise<GpuAdapterLike | null>;
          };
        }
      ).gpu;

      if (!gpu) return null;

      (
        ort.env.webgpu as unknown as {
          powerPreference?: 'low-power' | 'high-performance';
        }
      ).powerPreference = 'high-performance';

      const adapter = await gpu.requestAdapter({
        powerPreference: 'high-performance',
      });

      if (!adapter) {
        adapterProbe = null;
        return null;
      }

      if (!gpuDevice) {
        gpuDevice = await adapter.requestDevice();
      }

      return adapter;
    } catch {
      adapterProbe = null;
      return null;
    }
  })();

  return adapterProbe;
}

function watchGpuDeviceLoss(adapter: GpuAdapterLike) {
  if (deviceMonitorStarted) return;

  deviceMonitorStarted = true;

  const devicePromise = gpuDevice
    ? Promise.resolve(gpuDevice)
    : adapter.requestDevice();

  devicePromise
    .then((device) =>
      device.lost.then((info) => {
        deviceMonitorStarted = false;
        gpuDevice = null;

        if (gpuFailed) return;

        gpuFailed = true;
        postGpuStatus(false, `GPU device lost: ${info.reason ?? 'unknown'}`);
        void releaseSession();
      }),
    )
    .catch(() => {
      deviceMonitorStarted = false;
    });
}

function postGpuStatus(active: boolean, reason?: string) {
  if (!active && reason) {
    console.warn(`[worker] gpu inactive: ${reason}`);
  }

  self.postMessage({
    type: 'gpu-status',
    payload: { active, reason },
  });
}

async function releaseSession() {
  const previous = session;

  session = null;
  currentModelId = null;
  currentSource = null;
  currentGpu = false;

  if (!previous) return;

  try {
    await previous.release();
  } catch {
    /* session already gone */
  }
}

function configureWorker(numThreads: number, gpu?: boolean) {
  if (gpu !== undefined && gpu !== wantsGpu) {
    wantsGpu = gpu;

    if (gpu) {
      gpuFailed = false;
      gpuFailedModelId = null;
      resetAdapterProbe();
    }

    if (session) sessionStale = true;
  }

  const clamped = Math.max(
    1,
    Math.min(getMaxSessionThreads(), Math.floor(numThreads)),
  );

  if (clamped === sessionThreads) return;

  sessionThreads = clamped;

  if (!isElectron) {
    ort.env.wasm.numThreads = clamped;
  }

  if (session) sessionStale = true;
}

async function applyPendingThreadChange() {
  if (!sessionStale) return;

  const source = currentSource;

  if (!session || !source) {
    sessionStale = false;
    return;
  }

  const previous = session;
  const wantGpu = currentGpu && wantsGpu;

  let rebuilt: Awaited<ReturnType<typeof createSession>> | null = null;
  let rebuiltGpu = false;

  if (wantGpu) {
    try {
      rebuilt = await createSession(source, true);
      rebuiltGpu = true;
    } catch {
      rebuilt = null;
    }
  }

  if (!rebuilt) {
    try {
      rebuilt = await createSession(source, false);
      rebuiltGpu = false;
    } catch {
      rebuilt = null;
    }
  }

  if (!rebuilt) return;

  sessionStale = false;

  try {
    await previous.release();
  } catch {
    /* session already gone */
  }

  session = rebuilt;
  currentGpu = rebuiltGpu;
}

function postProgress(
  progress: number,
  status: string,
  tile?: { done: number; total: number },
  stage?: ModelLoadStage,
) {
  self.postMessage({
    type: 'progress',
    payload: { progress, status, jobId: currentJobId, tile, stage },
  });
}

async function createSession(
  source: string | ArrayBuffer | Uint8Array,
  gpu: boolean,
) {
  const executionProviders: ort.InferenceSession.SessionOptions['executionProviders'] =
    gpu && gpuPreferredLayout
      ? [{ name: 'webgpu', preferredLayout: gpuPreferredLayout }]
      : gpu
        ? ['webgpu']
        : ['wasm'];

  const options: ort.InferenceSession.SessionOptions = {
    executionProviders,
    graphOptimizationLevel: 'all',
    preferredOutputLocation: 'cpu',
  };

  if (typeof source === 'string') {
    return ort.InferenceSession.create(source, options);
  }

  const model =
    source instanceof Uint8Array ? source : new Uint8Array(source);

  return ort.InferenceSession.create(model, options);
}

async function warmupSession(
  target: Awaited<ReturnType<typeof createSession>>,
  tileSize: number,
) {
  const input = new ort.Tensor(
    'float32',
    new Float32Array(3 * tileSize * tileSize),
    [1, 3, tileSize, tileSize],
  );

  await target.run({ [target.inputNames[0]]: input });
}

async function loadModel(
  modelId: string,
  modelData?: ArrayBuffer,
  gpu = false,
  numThreads?: number,
  warmup = false,
  tileSize?: number,
  preferredLayout?: 'NCHW' | 'NHWC',
  gpuProbeTimeoutMs?: number,
) {
  try {
    if (numThreads != null) {
      configureWorker(numThreads, gpu);
    }

    wantsGpu = gpu;
    gpuPreferredLayout = preferredLayout;
    gpuInferTimeoutMs = gpuProbeTimeoutMs ?? null;

    const desiredTileSize =
      tileSize != null ? normalizeTileSize(tileSize) : currentTileSize;

    if (sessionStale) {
      sessionStale = false;

      if (session) {
        try {
          await session.release();
        } catch {
          /* session already gone */
        }

        session = null;
        currentModelId = null;
      }
    }

    if (gpuFailedModelId && gpuFailedModelId !== modelId) {
      gpuFailed = false;
      gpuFailedModelId = null;
    }

    if (
      session &&
      currentModelId === modelId &&
      currentGpu === gpu &&
      currentTileSize === desiredTileSize
    ) {
      self.postMessage({
        type: 'model-loaded',
        payload: { modelId, gpu },
      });

      return;
    }

    currentTileSize = desiredTileSize;

    postProgress(0, 'Loading model...', undefined, 'session');

    const sharedModel = providedModelBuffers.get(modelId);
    const source = sharedModel ?? modelData;

    console.log(
      `[worker] load-model id=${modelId} shared=${!!sharedModel} modelData=${!!modelData} bytes=${source?.byteLength ?? 0}`,
    );

    if (!source || source.byteLength === 0) {
      throw new Error(
        `No model data available. modelId=${modelId}, shared=${!!sharedModel}, modelData=${!!modelData}, bytes=${source?.byteLength ?? 0}`,
      );
    }

    let effectiveGpu = gpu;

    if (effectiveGpu) {
      const adapter = await probeWebGpuAdapter();

      if (!adapter || gpuFailed) {
        effectiveGpu = false;
        gpuFailed = true;
        gpuFailedModelId = modelId;

        postGpuStatus(
          false,
          adapter
            ? 'WebGPU unavailable for this model'
            : 'WebGPU needs a secure context (localhost or HTTPS)',
        );
      } else {
        watchGpuDeviceLoss(adapter);
      }
    }

    applyProvidedWasmBinary();

    const t0 = performance.now();

    console.log(
      `[worker] createSession start gpu=${effectiveGpu} bytes=${source.byteLength}`,
    );

    try {
      session = await createSession(source, effectiveGpu);
    } catch (gpuError) {
      console.warn(
        `[worker] gpu session failed (${(performance.now() - t0) | 0}ms): ${gpuError}`,
      );

      if (!effectiveGpu) throw gpuError;

      postProgress(0, 'Loading model...', undefined, 'fallback');

      session = await createSession(source, false);
      effectiveGpu = false;
      gpuFailed = true;
      gpuFailedModelId = modelId;

      postGpuStatus(false, String(gpuError));
    }

    currentSource = source;
    currentModelId = modelId;
    currentGpu = effectiveGpu;

    if (warmup && session) {
      try {
        await warmupSession(session, currentTileSize);
      } catch (warmError) {
        console.warn(`[worker] warmup failed: ${warmError}`);
      }
    }

    console.log(
      `[worker] model loaded in ${(performance.now() - t0).toFixed(0)}ms gpu=${effectiveGpu}`,
    );

    self.postMessage({
      type: 'model-loaded',
      payload: { modelId, gpu: effectiveGpu },
    });
  } catch (error) {
    const errorDetails =
      error instanceof Error
        ? `${error.name}: ${error.message}${error.stack ? `\n\nStack:\n${error.stack}` : ''}`
        : String(error);

    console.error('[worker] model load failed:', errorDetails);

    self.postMessage({
      type: 'error',
      payload: {
        message: `Failed to load model:\n${errorDetails}`,
        fatal: FATAL_PATTERN.test(errorDetails),
      },
    });
  }
}

function assertOutputSize(w: number, h: number) {
  if (
    w > MAX_OUTPUT_DIMENSION ||
    h > MAX_OUTPUT_DIMENSION ||
    w * h > MAX_OUTPUT_PIXELS
  ) {
    throw new Error(
      `Image too large: output would be ${w}×${h}px, exceeding the ` +
        `${MAX_OUTPUT_DIMENSION}px / ${Math.round(MAX_OUTPUT_PIXELS / 1_000_000)}MP limit`,
    );
  }
}

async function processImage(payload: {
  jobId: number;
  imageData: ImageData;
  alpha?: Uint8ClampedArray;
  width: number;
  height: number;
  scale: number;
  targetScale?: number;
}) {
  const { imageData, width, height, scale } = payload;
  const targetScale = payload.targetScale ?? scale;

  let pending: Promise<Float32Array> | undefined;

  try {
    await applyPendingThreadChange();

    if (!session) {
      throw new Error('Model not loaded');
    }

    currentJobId = payload.jobId;
    abortRequested = false;

    const rawFactor = scale / targetScale;
    const downsample = targetScale < scale && Number.isInteger(rawFactor);
    const blendScale = downsample ? targetScale : scale;

    const blendW = width * blendScale;
    const blendH = height * blendScale;
    const outW = width * targetScale;
    const outH = height * targetScale;

    assertOutputSize(blendW, blendH);

    if (outW !== blendW || outH !== blendH) {
      assertOutputSize(outW, outH);
    }

    postProgress(5, 'Preprocessing...');

    const outData = new Uint8ClampedArray(blendW * blendH * 4);

    const alphaPlane = payload.alpha
      ? scaleAlpha(payload.alpha, width, height, blendW, blendH)
      : null;

    const inputName = session.inputNames[0];
    const outputName = session.outputNames[0];

    const tileSize = currentTileSize;
    const tileStep = tileSize - TILE_OVERLAP;

    const tilesX = Math.max(1, Math.ceil((width - TILE_OVERLAP) / tileStep));
    const tilesY = Math.max(1, Math.ceil((height - TILE_OVERLAP) / tileStep));

    const tiles: Array<{ x0: number; y0: number; tx: number; ty: number }> = [];

    for (let ty = 0; ty < tilesY; ty++) {
      for (let tx = 0; tx < tilesX; tx++) {
        tiles.push({ x0: tx * tileStep, y0: ty * tileStep, tx, ty });
      }
    }

    const totalTiles = tiles.length;

    const buffers = [
      new Float32Array(3 * tileSize * tileSize),
      new Float32Array(3 * tileSize * tileSize),
    ];

    const weights = buildBlendWeights(blendScale, tileSize);

    let done = 0;
    let inferMs = 0;
    let cpuMs = 0;

    const startTile = (
      index: number,
      alreadyPacked = false,
    ): Promise<Float32Array> => {
      const buffer = buffers[index % 2];
      const { x0, y0 } = tiles[index];

      const refill = () =>
        extractTile(imageData, x0, y0, buffer, tileSize);

      if (!alreadyPacked) refill();

      const tensor = new ort.Tensor(
        'float32',
        buffer,
        [1, 3, tileSize, tileSize],
      );

      return runWithFallback(
        inputName,
        outputName,
        tensor,
        refill,
        (ms) => {
          inferMs += ms;
        },
      );
    };

    const jobStart = performance.now();

    pending = startTile(0);

    for (let i = 0; i < totalTiles; i++) {
      const { x0, y0, tx, ty } = tiles[i];

      postProgress(
        10 + Math.round((done / totalTiles) * 70),
        `Running inference... (${done + 1}/${totalTiles})`,
        { done, total: totalTiles },
      );

      const packStart = performance.now();

      if (i + 1 < totalTiles) {
        extractTile(
          imageData,
          tiles[i + 1].x0,
          tiles[i + 1].y0,
          buffers[(i + 1) % 2],
          tileSize,
        );
      }

      cpuMs += performance.now() - packStart;

      const tileOut = await pending;

      if (abortRequested) throw new AbortError();

      const blendStart = performance.now();

      const tileData = downsample
        ? downsampleTile(tileOut, tileSize * scale, rawFactor)
        : tileOut;

      const ax = tx === 0 ? weights.full : weights.ramp;
      const ay = ty === 0 ? weights.full : weights.ramp;

      blendTile(
        outData,
        tileData,
        x0,
        y0,
        blendW,
        blendH,
        blendScale,
        ax,
        ay,
        alphaPlane,
        tileSize,
      );

      cpuMs += performance.now() - blendStart;
      done++;

      if (i + 1 < totalTiles) {
        await yieldToEventLoop();

        if (abortRequested) throw new AbortError();

        pending = startTile(i + 1, true);
      }
    }

    console.log(
      `[worker] tiles=${totalTiles} ep=${currentGpu ? 'webgpu' : 'wasm'} ` +
        `inference=${inferMs | 0}ms cpu=${cpuMs | 0}ms wall=${(performance.now() - jobStart) | 0}ms`,
    );

    if (abortRequested) throw new AbortError();

    postProgress(85, 'Postprocessing...');

    const stitchedImageData = new ImageData(outData, blendW, blendH);

    let resultCanvas: OffscreenCanvas | HTMLCanvasElement =
      new OffscreenCanvas(blendW, blendH);

    resultCanvas.getContext('2d')!.putImageData(stitchedImageData, 0, 0);

    if (outW !== blendW || outH !== blendH) {
      const scaledCanvas = new OffscreenCanvas(outW, outH);
      const scaledCtx = scaledCanvas.getContext('2d')!;

      scaledCtx.imageSmoothingEnabled = true;
      scaledCtx.imageSmoothingQuality = 'high';
      scaledCtx.drawImage(resultCanvas, 0, 0, outW, outH);

      resultCanvas = scaledCanvas;
    }

    postProgress(92, 'Creating result...');

    const blob = await resultCanvas.convertToBlob({ type: 'image/png' });
    const resultUrl = URL.createObjectURL(blob);

    self.postMessage({
      type: 'complete',
      payload: {
        resultUrl,
        size: { width: outW, height: outH },
        jobId: currentJobId,
      },
    });
  } catch (error) {
    pending?.catch(() => undefined);

    if (error instanceof AbortError) {
      self.postMessage({
        type: 'aborted',
        payload: {
          message: 'Processing cancelled',
          jobId: currentJobId,
        },
      });
    } else {
      self.postMessage({
        type: 'error',
        payload: {
          message: `Processing failed: ${error}`,
          jobId: currentJobId,
          fatal: FATAL_PATTERN.test(String(error)),
        },
      });
    }
  } finally {
    abortRequested = false;
  }
}

function withTimeout<T>(
  promise: Promise<T>,
  ms: number,
  label: string,
): Promise<T> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      reject(new Error(`${label} timed out after ${Math.round(ms / 1000)}s`));
    }, ms);

    promise.then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      (error) => {
        clearTimeout(timer);
        reject(error);
      },
    );
  });
}

async function runWithFallback(
  inputName: string,
  outputName: string,
  tensor: ort.Tensor,
  refill: () => void,
  onRun?: (ms: number) => void,
): Promise<Float32Array> {
  try {
    const results = await timedRun(inputName, tensor, onRun);
    return results[outputName].data as Float32Array;
  } catch (error) {
    if (abortRequested) throw new AbortError();

    if (!currentGpu || !currentSource) throw error;

    self.postMessage({
      type: 'gpu-fallback',
      payload: {
        message: `GPU inference failed, falling back to CPU: ${error}`,
      },
    });

    postProgress(10, 'GPU failed, retrying on CPU...');

    session = await createSession(currentSource, false);
    currentGpu = false;
    gpuFailed = true;
    gpuFailedModelId = currentModelId;

    refill();

    const retry = await timedRun(inputName, tensor, onRun);

    return retry[outputName].data as Float32Array;
  }
}

async function timedRun(
  inputName: string,
  tensor: ort.Tensor,
  onRun?: (ms: number) => void,
) {
  const start = performance.now();

  const timeoutMs = currentGpu
    ? (gpuInferTimeoutMs ?? INFER_TIMEOUT_MS)
    : INFER_TIMEOUT_MS;

  try {
    return await withTimeout(
      session!.run({ [inputName]: tensor }),
      timeoutMs,
      'Inference',
    );
  } finally {
    onRun?.(performance.now() - start);
  }
}

function extractTile(
  imageData: ImageData,
  x0: number,
  y0: number,
  out: Float32Array,
  tileSize: number,
) {
  const { data, width, height } = imageData;
  const plane = tileSize * tileSize;
  const inv = 1 / 255;

  if (x0 + tileSize <= width && y0 + tileSize <= height) {
    for (let ly = 0; ly < tileSize; ly++) {
      let si = ((y0 + ly) * width + x0) * 4;
      let di = ly * tileSize;
      const diEnd = di + tileSize;

      for (; di < diEnd; di++, si += 4) {
        out[di] = data[si] * inv;
        out[plane + di] = data[si + 1] * inv;
        out[2 * plane + di] = data[si + 2] * inv;
      }
    }

    return;
  }

  for (let ly = 0; ly < tileSize; ly++) {
    const sy = Math.min(height - 1, Math.max(0, y0 + ly));
    const rowBase = sy * width;
    const outRow = ly * tileSize;

    for (let lx = 0; lx < tileSize; lx++) {
      const sx = Math.min(width - 1, Math.max(0, x0 + lx));
      const si = (rowBase + sx) * 4;
      const di = outRow + lx;

      out[di] = data[si] * inv;
      out[plane + di] = data[si + 1] * inv;
      out[2 * plane + di] = data[si + 2] * inv;
    }
  }
}

function buildBlendWeights(blendScale: number, tileSize: number) {
  const ow = tileSize * blendScale;
  const overlap = TILE_OVERLAP * blendScale;

  const full = new Float32Array(ow).fill(1);
  const ramp = new Float32Array(ow);

  for (let o = 0; o < ow; o++) {
    ramp[o] = Math.min(1, o / overlap);
  }

  return { full, ramp };
}

function downsampleTile(
  src: Float32Array,
  ow: number,
  factor: number,
): Float32Array {
  const owT = ow / factor;
  const planeSrc = ow * ow;
  const planeDst = owT * owT;

  const dst = new Float32Array(3 * planeDst);
  const inv = 1 / (factor * factor);

  for (let y = 0; y < owT; y++) {
    const sy0 = y * factor;

    for (let x = 0; x < owT; x++) {
      const sx0 = x * factor;

      let r = 0;
      let g = 0;
      let b = 0;

      for (let dy = 0; dy < factor; dy++) {
        let si = (sy0 + dy) * ow + sx0;

        for (let dx = 0; dx < factor; dx++, si++) {
          r += src[si];
          g += src[planeSrc + si];
          b += src[2 * planeSrc + si];
        }
      }

      const di = y * owT + x;

      dst[di] = r * inv;
      dst[planeDst + di] = g * inv;
      dst[2 * planeDst + di] = b * inv;
    }
  }

  return dst;
}

function scaleAlpha(
  alpha: Uint8ClampedArray,
  srcW: number,
  srcH: number,
  dstW: number,
  dstH: number,
): Uint8ClampedArray {
  if (srcW === dstW && srcH === dstH) return alpha;

  const source = new ImageData(srcW, srcH);

  for (let i = 0, p = 0; i < alpha.length; i++, p += 4) {
    source.data[p] = 255;
    source.data[p + 1] = 255;
    source.data[p + 2] = 255;
    source.data[p + 3] = alpha[i];
  }

  const srcCanvas = new OffscreenCanvas(srcW, srcH);
  srcCanvas.getContext('2d')!.putImageData(source, 0, 0);

  const dstCanvas = new OffscreenCanvas(dstW, dstH);
  const ctx = dstCanvas.getContext('2d')!;

  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(srcCanvas, 0, 0, dstW, dstH);

  const scaled = ctx.getImageData(0, 0, dstW, dstH).data;
  const out = new Uint8ClampedArray(dstW * dstH);

  for (let i = 0, p = 3; i < out.length; i++, p += 4) {
    out[i] = scaled[p];
  }

  return out;
}

function blendTile(
  out: Uint8ClampedArray,
  tile: Float32Array,
  x0: number,
  y0: number,
  destW: number,
  destH: number,
  blendScale: number,
  ax: Float32Array,
  ay: Float32Array,
  alphaPlane: Uint8ClampedArray | null,
  tileSize: number,
) {
  const ow = tileSize * blendScale;
  const plane = ow * ow;

  const baseX = x0 * blendScale;
  const baseY = y0 * blendScale;

  for (let oy = 0; oy < ow; oy++) {
    const gy = baseY + oy;

    if (gy >= destH) break;

    const aY = ay[oy];
    const tileRow = oy * ow;

    for (let ox = 0; ox < ow; ox++) {
      const gx = baseX + ox;

      if (gx >= destW) break;

      const ti = tileRow + ox;

      const r = tile[ti] * 255;
      const g = tile[plane + ti] * 255;
      const b = tile[2 * plane + ti] * 255;

      const di = (gy * destW + gx) * 4;

      const a = aY * ax[ox];

      if (a >= 1) {
        out[di] = r;
        out[di + 1] = g;
        out[di + 2] = b;
      } else {
        const inv = 1 - a;

        out[di] = out[di] * inv + r * a;
        out[di + 1] = out[di + 1] * inv + g * a;
        out[di + 2] = out[di + 2] * inv + b * a;
      }

      out[di + 3] = alphaPlane
        ? alphaPlane[gy * destW + gx]
        : 255;
    }
  }
}
