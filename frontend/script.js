const video = document.getElementById('videoElement');
const canvas = document.getElementById('canvasElement');
const predictionLabel = document.getElementById('predictionLabel');
const confidenceValue = document.getElementById('confidenceValue');
const connectionStatus = document.getElementById('connectionStatus');
const statusDot = document.querySelector('.status-dot');
const predictionDisplay = document.querySelector('.prediction-display');

const ctx = canvas.getContext('2d');
let ws;
let isStreaming = false;

// Connect to WebSocket
function connectWebSocket() {
    // Automatically use wss:// on HTTPS (production) and ws:// on HTTP (localhost)
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const wsUrl = `${protocol}//${host}/ws/predict`;
    
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        connectionStatus.textContent = 'Connected';
        statusDot.classList.add('connected');
        if (isStreaming) {
            sendFrames();
        }
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        if (data.error) {
            console.error('Backend Error:', data.error);
            return;
        }

        if (data.prediction !== "No Hand Detected") {
            // Only animate if the prediction changes
            if (predictionLabel.textContent !== data.prediction) {
                predictionDisplay.style.transform = 'scale(1.15)';
                setTimeout(() => {
                    predictionDisplay.style.transform = 'scale(1)';
                }, 200);
            }
            
            predictionLabel.textContent = data.prediction;
            confidenceValue.textContent = data.confidence;
            
        } else {
            predictionLabel.textContent = '--';
            confidenceValue.textContent = '0%';
        }
    };

    ws.onclose = () => {
        connectionStatus.textContent = 'Disconnected. Reconnecting...';
        statusDot.classList.remove('connected');
        predictionLabel.textContent = '--';
        confidenceValue.textContent = '0%';
        setTimeout(connectWebSocket, 3000); // Reconnect after 3 seconds
    };
    
    ws.onerror = (err) => {
        console.error('WebSocket Error:', err);
    };
}

// Start Webcam
async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: "user" 
            } 
        });
        video.srcObject = stream;
        
        video.onloadedmetadata = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            isStreaming = true;
            if (ws && ws.readyState === WebSocket.OPEN) {
                sendFrames();
            }
        };
    } catch (err) {
        console.error("Error accessing webcam:", err);
        connectionStatus.textContent = "Camera Error. Please allow access.";
    }
}

// Send frames continuously
function sendFrames() {
    if (!isStreaming || ws.readyState !== WebSocket.OPEN) return;

    // Draw video frame to canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Convert to JPEG base64
    // Using 0.5 quality to significantly reduce payload size and latency
    const base64Data = canvas.toDataURL('image/jpeg', 0.5);
    
    // Send to backend
    ws.send(base64Data);

    // Limit frame rate to ~15 FPS to avoid network congestion
    setTimeout(() => {
        requestAnimationFrame(sendFrames);
    }, 1000 / 15);
}

// Initialize
connectWebSocket();
startCamera();
