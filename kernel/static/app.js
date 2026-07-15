const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const statusDiv = document.getElementById('status');
const logDiv = document.getElementById('log');

let audioContext, workletNode, websocket, isRunning = false;

function log(msg) {
    logDiv.textContent = `${msg}\n${logDiv.textContent}`;
}

async function startStreaming() {
    try {
        statusDiv.textContent = "Status: Requesting microphone...";
        const stream = await navigator.mediaDevices.getUserMedia({ 
            audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true } 
        });

        audioContext = new AudioContext({ sampleRate: 16000 });
        await audioContext.audioWorklet.addModule('audio-worklet.js');
        
        workletNode = new AudioWorkletNode(audioContext, 'audio-streamer-processor');
        audioContext.createMediaStreamSource(stream).connect(workletNode);

        websocket = new WebSocket('ws://localhost:8765');
        websocket.binaryType = 'arraybuffer';

        websocket.onopen = () => {
            statusDiv.textContent = "Status: Streaming";
            isRunning = true;
            startBtn.disabled = true;
            stopBtn.disabled = false;
        };

        websocket.onmessage = (event) => {
            if (event.data.startsWith('DETECTED:')) {
                log(`>>> ${event.data.split(':')[1].toUpperCase()} <<<`);
            }
        };

        websocket.onclose = () => {
            statusDiv.textContent = "Status: Disconnected";
            isRunning = false;
            startBtn.disabled = false;
            stopBtn.disabled = true;
        };

        workletNode.port.onmessage = (event) => {
            if (isRunning && websocket.readyState === WebSocket.OPEN) {
                websocket.send(event.data);
            }
        };
    } catch (err) {
        statusDiv.textContent = `Error: ${err.message}`;
    }
}

function stopStreaming() {
    isRunning = false;
    if (websocket) websocket.close();
    if (audioContext) audioContext.close();
    startBtn.disabled = false;
    stopBtn.disabled = true;
    statusDiv.textContent = "Status: Idle";
}

startBtn.addEventListener('click', startStreaming);
stopBtn.addEventListener('click', stopStreaming);
