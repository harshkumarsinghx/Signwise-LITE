const video = document.getElementById('videoElement');
const liveCanvas = document.getElementById('liveCanvas');
const captureCanvas = document.getElementById('captureCanvas');
const predictionLabel = document.getElementById('predictionLabel');
const confidenceValue = document.getElementById('confidenceValue');
const connectionStatus = document.getElementById('connectionStatus');
const statusDot = document.querySelector('.status-dot');
const predictionDisplay = document.querySelector('.prediction-display');

const liveCtx = liveCanvas.getContext('2d');
const captureCtx = captureCanvas.getContext('2d');

let ws;
let isStreaming = false;
let currentLandmarks = null;

// ── Prediction Smoothing ──────────────────────────────────────────────────
let predictionHistory = [];

function updatePrediction(prediction, confidence) {
    if (prediction === 'No Hand Detected') {
        predictionLabel.textContent = '--';
        confidenceValue.textContent = '0%';
        predictionHistory = []; // Reset
        return;
    }

    predictionHistory.push(prediction);
    if (predictionHistory.length > 5) predictionHistory.shift(); // Keep last 5 frames

    // Get most frequent prediction (mode)
    const counts = {};
    let maxCount = 0;
    let mode = prediction;
    for (const p of predictionHistory) {
        counts[p] = (counts[p] || 0) + 1;
        if (counts[p] > maxCount) {
            maxCount = counts[p];
            mode = p;
        }
    }

    if (predictionLabel.textContent !== mode) {
        predictionDisplay.style.transform = 'scale(1.15)';
        setTimeout(() => { predictionDisplay.style.transform = 'scale(1)'; }, 200);
    }
    
    predictionLabel.textContent = mode;
    confidenceValue.textContent = confidence;
}

// ── MediaPipe Hand Connections (21 landmarks, 0-indexed) ─────────────────
const HAND_CONNECTIONS = [
    [0, 1],  [1, 2],   [2, 3],   [3, 4],   // Thumb
    [0, 5],  [5, 6],   [6, 7],   [7, 8],   // Index
    [5, 9],  [9, 10],  [10, 11], [11, 12], // Middle
    [9, 13], [13, 14], [14, 15], [15, 16], // Ring
    [13, 17],[17, 18], [18, 19], [19, 20], // Pinky
    [0, 17]                                 // Palm base
];

const FINGERTIP_IDS = new Set([4, 8, 12, 16, 20]);

// ── Draw skeleton ────────────────────────────────────────────────────────
function drawLandmarks(landmarks, w, h) {
    const pts = landmarks.map(([x, y]) => [x * w, y * h]);

    // Draw connections
    liveCtx.lineWidth = 3;
    liveCtx.lineCap = 'round';

    for (const [a, b] of HAND_CONNECTIONS) {
        const grad = liveCtx.createLinearGradient(pts[a][0], pts[a][1], pts[b][0], pts[b][1]);
        grad.addColorStop(0, '#3b82f6');
        grad.addColorStop(1, '#8b5cf6');
        liveCtx.strokeStyle = grad;

        liveCtx.beginPath();
        liveCtx.moveTo(pts[a][0], pts[a][1]);
        liveCtx.lineTo(pts[b][0], pts[b][1]);
        liveCtx.stroke();
    }

    // Draw dots
    pts.forEach(([x, y], i) => {
        const isTip = FINGERTIP_IDS.has(i);

        // Outer glow
        liveCtx.beginPath();
        liveCtx.arc(x, y, isTip ? 8 : 5, 0, Math.PI * 2);
        liveCtx.fillStyle = isTip ? 'rgba(139, 92, 246, 0.4)' : 'rgba(59, 130, 246, 0.3)';
        liveCtx.fill();

        // Core dot
        liveCtx.beginPath();
        liveCtx.arc(x, y, isTip ? 5 : 3, 0, Math.PI * 2);
        liveCtx.fillStyle = isTip ? '#c084fc' : '#60a5fa';
        liveCtx.fill();
    });
}

// ── Render Loop ──────────────────────────────────────────────────────────
function renderLoop() {
    if (isStreaming) {
        const w = liveCanvas.width;
        const h = liveCanvas.height;

        liveCtx.save();
        
        // Mirror the canvas context so video and skeleton appear mirrored
        liveCtx.translate(w, 0);
        liveCtx.scale(-1, 1);

        // Draw current video frame
        liveCtx.drawImage(video, 0, 0, w, h);

        // Draw latest landmarks on top
        if (currentLandmarks && currentLandmarks.length > 0) {
            drawLandmarks(currentLandmarks, w, h);
        }

        liveCtx.restore();
    }
    requestAnimationFrame(renderLoop);
}

// ── WebSocket ─────────────────────────────────────────────────────────────
function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const wsUrl = `${protocol}//${host}/ws/predict`;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        connectionStatus.textContent = 'Connected';
        statusDot.classList.add('connected');
        if (isStreaming) sendFrames();
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.error) {
            console.error('Backend Error:', data.error);
            return;
        }

        currentLandmarks = data.landmarks || null;
        updatePrediction(data.prediction, data.confidence);
    };

    ws.onclose = () => {
        connectionStatus.textContent = 'Disconnected. Reconnecting…';
        statusDot.classList.remove('connected');
        currentLandmarks = null;
        updatePrediction('No Hand Detected', '0%');
        setTimeout(connectWebSocket, 3000);
    };

    ws.onerror = (err) => console.error('WebSocket Error:', err);
}

// ── Camera ────────────────────────────────────────────────────────────────
async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }
        });
        video.srcObject = stream;

        video.onloadedmetadata = () => {
            liveCanvas.width = video.videoWidth;
            liveCanvas.height = video.videoHeight;
            captureCanvas.width = video.videoWidth;
            captureCanvas.height = video.videoHeight;

            isStreaming = true;
            if (ws && ws.readyState === WebSocket.OPEN) sendFrames();
            
            // Start continuous rendering loop
            requestAnimationFrame(renderLoop);
        };
    } catch (err) {
        console.error('Error accessing webcam:', err);
        connectionStatus.textContent = 'Camera Error. Please allow access.';
    }
}

// ── Frame sender ──────────────────────────────────────────────────────────
function sendFrames() {
    if (!isStreaming || ws.readyState !== WebSocket.OPEN) return;

    // Capture frame on the hidden, unmirrored canvas to send to backend
    captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
    const base64Data = captureCanvas.toDataURL('image/jpeg', 0.5);
    ws.send(base64Data);

    // Limit to ~15 FPS to avoid network overload
    setTimeout(() => sendFrames(), 1000 / 15);
}

// ── Init ──────────────────────────────────────────────────────────────────
connectWebSocket();
startCamera();
