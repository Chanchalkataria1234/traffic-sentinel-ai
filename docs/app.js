// Configure ONNX Runtime Web to load WebAssembly binaries from CDN
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/';

// Class Lists
const COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
];

const INDIAN_VEHICLE_CLASSES = [
    'ambulance', 'army vehicle', 'auto rickshaw', 'bicycle', 'bus', 'car', 'garbagevan', 'human hauler', 
    'minibus', 'minivan', 'motorbike', 'pickup', 'policecar', 'rickshaw', 'scooter', 'suv', 'taxi', 
    'three wheelers -CNG-', 'truck', 'van', 'wheelbarrow'
];

// Color mapping for classes
function getClassColor(classId) {
    const colors = [
        '#3b82f6', '#10b981', '#8b5cf6', '#f43f5e', '#f59e0b',
        '#06b6d4', '#ec4899', '#14b8a6', '#f97316', '#a855f7',
        '#eab308', '#6366f1', '#a3e635', '#f472b6', '#38bdf8',
        '#fb7185', '#22c55e', '#a855f7', '#3b82f6', '#06b6d4'
    ];
    return colors[classId % colors.length];
}

// State variables
let session = null;
let isLoaded = false;
let classNames = [...INDIAN_VEHICLE_CLASSES];
let currentMode = 'image'; // 'image', 'video', 'webcam'
let webcamStream = null;
let isProcessingFrame = false;
let animationFrameId = null;

// Active data for canvas drawing & highlights
let currentDetections = [];
let currentPrep = null;

// DOM Elements
const workspaceActions = document.getElementById('workspace-actions');
const btnSubmit = document.getElementById('btn-submit');
const btnClear = document.getElementById('btn-clear');

const confThreshold = document.getElementById('conf-threshold');
const confVal = document.getElementById('conf-val');
const iouThreshold = document.getElementById('iou-threshold');
const iouVal = document.getElementById('iou-val');

const modeTabs = document.querySelectorAll('.mode-tab');
const workspacePanels = document.querySelectorAll('.workspace-panel');

const imageDropZone = document.getElementById('image-drop-zone');
const imageFileInput = document.getElementById('image-file-input');
const videoDropZone = document.getElementById('video-drop-zone');
const videoFileInput = document.getElementById('video-file-input');

const canvasContainer = document.getElementById('canvas-container');
const outputCanvas = document.getElementById('output-canvas');
const hiddenVideo = document.getElementById('hidden-video');

const videoControls = document.getElementById('video-controls');
const btnVideoPlay = document.getElementById('btn-video-play');
const videoProgress = document.getElementById('video-progress');
const videoTime = document.getElementById('video-time');

const webcamPlaceholder = document.querySelector('.webcam-placeholder');
const btnWebcamToggle = document.getElementById('btn-webcam-toggle');

const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');
const inferenceTimeBadge = document.getElementById('inference-time-badge');
const statsTotalCount = document.getElementById('stats-total-count');
const classSummaryContainer = document.getElementById('class-summary-container');
const detectionsTableContainer = document.getElementById('detections-table-container');
const detectionsTableBody = document.getElementById('detections-table-body');
const inferenceLoading = document.getElementById('inference-loading');

// Initialize UI States
document.addEventListener('DOMContentLoaded', () => {
    // Setup event listeners
    setupEventListeners();
    
    // Load default model
    loadModel();
});

// Event Listeners Configuration
function setupEventListeners() {
    // Submit and Clear Actions
    if (btnSubmit) {
        btnSubmit.addEventListener('click', () => {
            if (currentMode === 'image') {
                submitImageDetection();
            }
        });
    }
    if (btnClear) {
        btnClear.addEventListener('click', () => {
            clearWorkspace();
        });
    }

    // Threshold sliders
    confThreshold.addEventListener('input', (e) => {
        confVal.textContent = parseFloat(e.target.value).toFixed(2);
        triggerReinference();
    });
    iouThreshold.addEventListener('input', (e) => {
        iouVal.textContent = parseFloat(e.target.value).toFixed(2);
        triggerReinference();
    });

    // No model selectors needed. Loaded automatically on startup.

    // Mode / Tab selector switches
    modeTabs.forEach(tab => {
        tab.addEventListener('click', (e) => {
            const tabEl = e.currentTarget;
            const mode = tabEl.dataset.mode;
            switchMode(mode);
        });
    });

    // Image Upload Click & Drag/Drop
    imageDropZone.addEventListener('click', () => imageFileInput.click());
    imageFileInput.addEventListener('change', handleImageUpload);
    setupDragDrop(imageDropZone, handleImageFile);

    // Video Upload Click & Drag/Drop
    videoDropZone.addEventListener('click', () => videoFileInput.click());
    videoFileInput.addEventListener('change', handleVideoUpload);
    setupDragDrop(videoDropZone, handleVideoFile);

    // Video control play/pause
    btnVideoPlay.addEventListener('click', toggleVideoPlay);
    hiddenVideo.addEventListener('timeupdate', updateVideoProgress);
    hiddenVideo.addEventListener('ended', stopVideoLoop);

    // Webcam button control
    btnWebcamToggle.addEventListener('click', toggleWebcam);
}

// Switch between Image, Video, and Webcam modes
function switchMode(mode) {
    if (mode === currentMode) return;
    
    // Stop active streams, clear canvases
    clearWorkspace();
    
    currentMode = mode;
    
    // Update active tab styling
    modeTabs.forEach(tab => {
        if (tab.dataset.mode === mode) {
            tab.classList.add('active');
        } else {
            tab.classList.remove('active');
        }
    });

    // Toggle panel visibility
    workspacePanels.forEach(panel => {
        if (panel.id === `workspace-${mode}`) {
            panel.classList.remove('hidden');
        } else {
            panel.classList.add('hidden');
        }
    });

    // Reset workspace display and stats
    canvasContainer.classList.add('hidden');
    if (workspaceActions) workspaceActions.classList.add('hidden');
    videoControls.classList.add('hidden');
    resetResults();
    
    // Re-render placeholders if needed
    if (mode === 'webcam') {
        webcamPlaceholder.classList.remove('hidden');
    }
}

// Cleanup active cameras/animations when switching tabs
function cleanupActiveModes() {
    stopVideoLoop();
    hiddenVideo.src = '';
    hiddenVideo.load();
    
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }
    
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
    isProcessingFrame = false;
}

// Setup drag and drop events
function setupDragDrop(element, fileHandler) {
    ['dragenter', 'dragover'].forEach(eventName => {
        element.addEventListener(eventName, (e) => {
            e.preventDefault();
            element.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        element.addEventListener(eventName, (e) => {
            e.preventDefault();
            element.classList.remove('dragover');
        }, false);
    });

    element.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const file = dt.files[0];
        if (file) {
            fileHandler(file);
        }
    }, false);
}

// Helper to update status bar
function updateStatus(text, state = 'idle') {
    statusText.textContent = text;
    statusDot.className = 'status-dot';
    if (state === 'active') statusDot.classList.add('active');
    else if (state === 'loading') statusDot.classList.add('loading');
}

// Helper to set model loading spinner
function setLoadingState(loading, text = "") {
    if (loading) {
        if (text) updateStatus(text, 'loading');
    }
}

// Load ONNX Model session
async function loadModel() {
    setLoadingState(true, "Initializing ONNX Model...");
    
    try {
        if (session) {
            session = null;
            isLoaded = false;
        }

        // Fetch `./best.onnx` from the deployed root folder
        const modelInput = './best.onnx';
        updateStatus('Loading custom vehicle detection model (best.onnx)...', 'loading');

        // Initialize session (defaulting to WASM, which is supported on all browsers)
        session = await ort.InferenceSession.create(modelInput, {
            executionProviders: ['wasm']
        });

        isLoaded = true;
        updateStatus('Model loaded & ready', 'active');
        
        // If we loaded a model and already have an image loaded, run inference again
        triggerReinference();
        
    } catch (err) {
        console.error("ONNX model load failed:", err);
        setLoadingState(false);
        updateStatus('Load failed', 'idle');
        
        let errorMsg = `Failed to load model: ${err.message}\n\n`;
        if (source === 'custom') {
            errorMsg += "Please ensure you have generated your custom ONNX model (runs/detect/train6/weights/best.onnx) and copied it into the 'web_ui' directory as 'best.onnx'.\n\nAlternatively, select 'Upload Custom .onnx Model' to load the file directly from your disk.";
        }
        alert(errorMsg);
    }
}

function readFileAsArrayBuffer(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = () => reject(reader.error);
        reader.readAsArrayBuffer(file);
    });
}

// Preprocessing: letterboxing the image/video frame into a 640x640 Float32 tensor
function preprocess(source, targetWidth = 640, targetHeight = 640) {
    const canvas = document.createElement('canvas');
    canvas.width = targetWidth;
    canvas.height = targetHeight;
    const ctx = canvas.getContext('2d');

    // Fill background with YOLOv8 default gray (114, 114, 114)
    ctx.fillStyle = 'rgb(114, 114, 114)';
    ctx.fillRect(0, 0, targetWidth, targetHeight);

    // Get input dimensions
    const origW = source.naturalWidth || source.videoWidth || source.width || 640;
    const origH = source.naturalHeight || source.videoHeight || source.height || 640;

    // Calculate aspect ratio scale and borders (letterboxing)
    const scale = Math.min(targetWidth / origW, targetHeight / origH);
    const newW = origW * scale;
    const newH = origH * scale;
    const dx = (targetWidth - newW) / 2;
    const dy = (targetHeight - newH) / 2;

    // Draw frame centered
    ctx.drawImage(source, dx, dy, newW, newH);

    // Retrieve canvas pixel RGBA data
    const imgData = ctx.getImageData(0, 0, targetWidth, targetHeight);
    const data = imgData.data;

    // Rearrange from RGBA to Float32 CHW Planar format [1, 3, 640, 640]
    const float32Data = new Float32Array(3 * targetWidth * targetHeight);
    const imageSize = targetWidth * targetHeight;

    for (let i = 0; i < imageSize; i++) {
        const r = data[i * 4] / 255.0;
        const g = data[i * 4 + 1] / 255.0;
        const b = data[i * 4 + 2] / 255.0;

        float32Data[i] = r;                  // Red Planar
        float32Data[imageSize + i] = g;      // Green Planar
        float32Data[2 * imageSize + i] = b;  // Blue Planar
    }

    return {
        tensor: new ort.Tensor('float32', float32Data, [1, 3, targetWidth, targetHeight]),
        scale: scale,
        dx: dx,
        dy: dy,
        origW: origW,
        origH: origH
    };
}

// Bounding Box IoU calculation
function calculateIoU(box1, box2) {
    const [x1_1, y1_1, x2_1, y2_1] = box1;
    const [x1_2, y1_2, x2_2, y2_2] = box2;

    const x_left = Math.max(x1_1, x1_2);
    const y_top = Math.max(y1_1, y1_2);
    const x_right = Math.min(x2_1, x2_2);
    const y_bottom = Math.min(y2_1, y2_2);

    if (x_right < x_left || y_bottom < y_top) {
        return 0.0;
    }

    const intersectionArea = (x_right - x_left) * (y_bottom - y_top);
    const area1 = (x2_1 - x1_1) * (y2_1 - y1_1);
    const area2 = (x2_2 - x1_2) * (y2_2 - y1_2);

    const unionArea = area1 + area2 - intersectionArea;
    if (unionArea <= 0) return 0.0;
    
    return intersectionArea / unionArea;
}

// Non-Maximum Suppression (NMS)
function nonMaxSuppression(boxes, confidences, classIds, iouThreshold) {
    const indices = Array.from(confidences.keys());
    
    // Sort indices by score descending
    indices.sort((a, b) => confidences[b] - confidences[a]);

    const keep = [];
    const active = new Array(indices.length).fill(true);

    for (let i = 0; i < indices.length; i++) {
        if (!active[i]) continue;

        const idx = indices[i];
        keep.push(idx);

        for (let j = i + 1; j < indices.length; j++) {
            if (!active[j]) continue;

            const idx2 = indices[j];
            
            // Per-class suppression (only suppress overlapping boxes of the same class)
            if (classIds[idx] === classIds[idx2]) {
                const iou = calculateIoU(boxes[idx], boxes[idx2]);
                if (iou > iouThreshold) {
                    active[j] = false;
                }
            }
        }
    }
    return keep;
}

// Post-processing raw model output (shape [1, 4 + C, 8400])
function postprocess(outputData, prep) {
    const confThresh = parseFloat(confThreshold.value);
    const iouThresh = parseFloat(iouThreshold.value);

    // Auto-detect number of attributes (4 coordinates + C classes)
    const numAnchors = 8400;
    const numAttributes = outputData.length / numAnchors; 
    const numClasses = numAttributes - 4;

    const boxes = [];
    const confidences = [];
    const classIds = [];

    // Parse each anchor
    for (let j = 0; j < numAnchors; j++) {
        // Find best class score
        let maxScore = -Infinity;
        let maxClassId = -1;
        for (let c = 0; c < numClasses; c++) {
            const score = outputData[(4 + c) * numAnchors + j];
            if (score > maxScore) {
                maxScore = score;
                maxClassId = c;
            }
        }

        if (maxScore > confThresh) {
            // Coordinate points center_x, center_y, width, height (scaled [0, 640])
            const cx = outputData[0 * numAnchors + j];
            const cy = outputData[1 * numAnchors + j];
            const w = outputData[2 * numAnchors + j];
            const h = outputData[3 * numAnchors + j];

            // Convert to top-left and bottom-right coords
            const x1 = cx - w / 2;
            const y1 = cy - h / 2;
            const x2 = cx + w / 2;
            const y2 = cy + h / 2;

            boxes.push([x1, y1, x2, y2]);
            confidences.push(maxScore);
            classIds.push(maxClassId);
        }
    }

    // Execute NMS
    const keepIndices = nonMaxSuppression(boxes, confidences, classIds, iouThresh);

    // Map selected indices to final results, restoring original coordinates
    const finalDetections = [];
    for (let i = 0; i < keepIndices.length; i++) {
        const idx = keepIndices[i];
        const [x1, y1, x2, y2] = boxes[idx];

        // Map back from letterboxed padded space
        let orig_x1 = (x1 - prep.dx) / prep.scale;
        let orig_y1 = (y1 - prep.dy) / prep.scale;
        let orig_x2 = (x2 - prep.dx) / prep.scale;
        let orig_y2 = (y2 - prep.dy) / prep.scale;

        // Clip to image boundary
        orig_x1 = Math.max(0, Math.min(prep.origW, orig_x1));
        orig_y1 = Math.max(0, Math.min(prep.origH, orig_y1));
        orig_x2 = Math.max(0, Math.min(prep.origW, orig_x2));
        orig_y2 = Math.max(0, Math.min(prep.origH, orig_y2));

        finalDetections.push({
            box: [Math.round(orig_x1), Math.round(orig_y1), Math.round(orig_x2), Math.round(orig_y2)],
            score: confidences[idx],
            classId: classIds[idx]
        });
    }

    return finalDetections;
}

// Run Inference on any media source (img or video frame)
async function runInference(mediaSource) {
    if (!session) return null;
    
    // 1. Preprocess
    const prep = preprocess(mediaSource);
    
    // 2. Load into Float32 ONNX tensor and execute session
    const inputs = {};
    inputs[session.inputNames[0]] = prep.tensor;
    
    const outputMap = await session.run(inputs);
    const outputTensor = outputMap[session.outputNames[0]];
    
    // 3. Postprocess
    const detections = postprocess(outputTensor.data, prep);
    
    return {
        detections: detections,
        prep: prep
    };
}

// Redraw canvas with all detections
function renderDetections(source, results, inferenceTime) {
    currentDetections = results.detections;
    currentPrep = results.prep;

    // Display Canvas elements
    const workspacePanel = document.getElementById(`workspace-${currentMode}`);
    workspacePanel.classList.add('hidden');
    canvasContainer.classList.remove('hidden');

    outputCanvas.width = currentPrep.origW;
    outputCanvas.height = currentPrep.origH;

    const ctx = outputCanvas.getContext('2d');
    ctx.drawImage(source, 0, 0, currentPrep.origW, currentPrep.origH);

    // Draw boxes
    currentDetections.forEach((det, idx) => {
        drawBoundingBox(ctx, det, idx, false);
    });

    // Update Results Pane
    updateResultsUI(currentDetections, inferenceTime);
}

// Draw a single bounding box with styling
function drawBoundingBox(ctx, det, index, isHighlighted = false) {
    const [x1, y1, x2, y2] = det.box;
    const score = det.score;
    const classId = det.classId;
    const className = classNames[classId] || `Class ${classId}`;
    const color = getClassColor(classId);

    // Bounding Box stroke configuration
    ctx.strokeStyle = color;
    ctx.lineWidth = isHighlighted 
        ? Math.max(5, Math.round(ctx.canvas.width / 120)) 
        : Math.max(2, Math.round(ctx.canvas.width / 250));
    ctx.lineJoin = 'round';
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    // Shadow glows for highlights
    if (isHighlighted) {
        ctx.shadowBlur = 15;
        ctx.shadowColor = color;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        ctx.shadowBlur = 0; // reset
    }

    // Label Text configuration
    const fontSize = Math.max(12, Math.round(ctx.canvas.width / 50));
    ctx.font = `600 ${fontSize}px Inter, sans-serif`;
    const labelText = `${className} ${(score * 100).toFixed(0)}%`;
    const textWidth = ctx.measureText(labelText).width;
    const padding = 6;
    const textHeight = fontSize;

    // Put text background on top, or pull inside if at topmost boundary
    const textY = y1 - textHeight - (padding * 2) > 0 ? y1 - textHeight - (padding * 2) : y1;

    ctx.fillStyle = color;
    ctx.fillRect(x1 - (ctx.lineWidth / 2), textY, textWidth + (padding * 2), textHeight + (padding * 2));

    ctx.fillStyle = '#ffffff';
    ctx.fillText(labelText, x1 + padding - (ctx.lineWidth / 2), textY + textHeight + padding - 2);
}

// Bounding box overlay highlight during hover in results table
function highlightDetection(index) {
    if (!currentPrep) return;
    const ctx = outputCanvas.getContext('2d');
    
    // Clear canvas and draw source image
    const source = getActiveImageSource();
    if (!source) return;
    
    ctx.drawImage(source, 0, 0, currentPrep.origW, currentPrep.origH);

    // Draw normal boxes
    currentDetections.forEach((det, idx) => {
        if (idx !== index) {
            drawBoundingBox(ctx, det, idx, false);
        }
    });

    // Draw highlighted box last so it stacks on top
    if (index !== -1 && currentDetections[index]) {
        drawBoundingBox(ctx, currentDetections[index], index, true);
    }
}

// Helper to locate active image/video sources in document
function getActiveImageSource() {
    if (currentMode === 'image') {
        const loadedImg = document.getElementById('loaded-image');
        return loadedImg;
    } else if (currentMode === 'video' || currentMode === 'webcam') {
        return hiddenVideo;
    }
    return null;
}

// Update results layout metrics
function updateResultsUI(detections, inferenceTime) {
    // Inference speed badge
    inferenceTimeBadge.textContent = `Inference: ${inferenceTime}ms`;
    inferenceTimeBadge.classList.remove('hidden');

    // Total Count
    statsTotalCount.textContent = detections.length;

    // Calculate distributions
    const counts = {};
    detections.forEach(det => {
        const className = classNames[det.classId] || `Class ${det.classId}`;
        counts[className] = (counts[className] || 0) + 1;
    });

    // Render summary badges
    classSummaryContainer.innerHTML = '';
    if (detections.length === 0) {
        classSummaryContainer.innerHTML = '<div class="no-detections-text">No vehicles detected in this frame.</div>';
        detectionsTableContainer.classList.add('hidden');
        return;
    }

    Object.keys(counts).forEach(key => {
        const badge = document.createElement('div');
        badge.className = 'class-badge';
        badge.innerHTML = `
            <span>${key}</span>
            <span class="class-badge-count">${counts[key]}</span>
        `;
        classSummaryContainer.appendChild(badge);
    });

    // Render detailed coordinates table
    detectionsTableBody.innerHTML = '';
    detections.forEach((det, idx) => {
        const row = document.createElement('tr');
        const className = classNames[det.classId] || `Class ${det.classId}`;
        const color = getClassColor(det.classId);
        
        row.innerHTML = `
            <td>${idx + 1}</td>
            <td><span class="td-badge" style="background-color: ${color}20; color: ${color}; border: 1px solid ${color}40">${className}</span></td>
            <td style="font-family: monospace; font-weight: 500">${(det.score * 100).toFixed(1)}%</td>
            <td style="font-family: monospace; color: var(--text-secondary)">[${det.box.join(', ')}]</td>
        `;

        // Row hover highlight events
        row.addEventListener('mouseenter', () => {
            row.classList.add('table-row-highlight');
            highlightDetection(idx);
        });
        row.addEventListener('mouseleave', () => {
            row.classList.remove('table-row-highlight');
            highlightDetection(-1);
        });

        detectionsTableBody.appendChild(row);
    });
    
    detectionsTableContainer.classList.remove('hidden');
}

// Reset results widgets back to idle
function resetResults() {
    inferenceTimeBadge.classList.add('hidden');
    statsTotalCount.textContent = '0';
    classSummaryContainer.innerHTML = '<div class="no-detections-text">No vehicles detected yet. Upload an image or start the camera feed to begin.</div>';
    detectionsTableContainer.classList.add('hidden');
    detectionsTableBody.innerHTML = '';
    currentDetections = [];
    currentPrep = null;
}

// Retrigger inference when threshold sliders move (images only)
async function triggerReinference() {
    if (currentMode === 'image') {
        const loadedImg = document.getElementById('loaded-image');
        if (loadedImg && isLoaded) {
            inferenceLoading.classList.remove('hidden');
            const startTime = performance.now();
            const results = await runInference(loadedImg);
            const duration = Math.round(performance.now() - startTime);
            inferenceLoading.classList.add('hidden');
            
            if (results) {
                renderDetections(loadedImg, results, duration);
            }
        }
    }
}

// Handle image uploads
function handleImageUpload(e) {
    const file = e.target.files[0];
    if (file) {
        handleImageFile(file);
    }
}

function handleImageFile(file) {
    if (!file.type.startsWith('image/')) {
        alert("Please upload a valid image file.");
        return;
    }

    resetResults();
    
    const reader = new FileReader();
    reader.onload = (event) => {
        // Clean up previous image element if it exists
        const oldImg = document.getElementById('loaded-image');
        if (oldImg) oldImg.remove();
        
        const img = new Image();
        img.id = 'loaded-image';
        img.src = event.target.result;
        img.className = 'hidden';
        
        img.onload = () => {
            document.getElementById('workspace-image').appendChild(img);
            
            // Size the canvas to the loaded image dimensions and draw preview
            outputCanvas.width = img.naturalWidth;
            outputCanvas.height = img.naturalHeight;
            const ctx = outputCanvas.getContext('2d');
            ctx.drawImage(img, 0, 0);
            
            // Hide the upload panel and show canvas
            const workspacePanel = document.getElementById('workspace-image');
            workspacePanel.classList.add('hidden');
            canvasContainer.classList.remove('hidden');
            
            // Show action buttons
            updateActionsBar();
            
            if (!isLoaded) {
                updateStatus('Image loaded. Waiting for model to initialize...', 'loading');
            } else {
                updateStatus('Image loaded. Click "Submit Scan" to start vehicle detection.', 'active');
            }
        };
    };
    reader.readAsDataURL(file);
}

// Handle video uploads
function handleVideoUpload(e) {
    const file = e.target.files[0];
    if (file) {
        handleVideoFile(file);
    }
}

function handleVideoFile(file) {
    if (!file.type.startsWith('video/')) {
        alert("Please upload a valid video file.");
        return;
    }

    resetResults();
    cleanupActiveModes();

    hiddenVideo.onloadedmetadata = () => {
        // Prepare video panel workspace
        const workspacePanel = document.getElementById('workspace-video');
        workspacePanel.classList.add('hidden');
        
        canvasContainer.classList.remove('hidden');
        videoControls.classList.remove('hidden');
        
        // Show action buttons (only Clear Workspace will be visible in video mode)
        updateActionsBar();
        
        outputCanvas.width = hiddenVideo.videoWidth;
        outputCanvas.height = hiddenVideo.videoHeight;
        
        const ctx = outputCanvas.getContext('2d');
        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, outputCanvas.width, outputCanvas.height);
        
        updateVideoTime(0, hiddenVideo.duration);
        updateStatus('Video loaded. Press Play to start vehicle detection.', 'active');
    };

    const fileURL = URL.createObjectURL(file);
    hiddenVideo.src = fileURL;
    hiddenVideo.load();
}

// Video playback controls
let isVideoPlaying = false;

function toggleVideoPlay() {
    if (!hiddenVideo.src) return;
    
    if (!isLoaded) {
        alert("Please initialize the model before running detections on video.");
        return;
    }

    if (isVideoPlaying) {
        hiddenVideo.pause();
        stopVideoLoop();
    } else {
        hiddenVideo.play();
        isVideoPlaying = true;
        document.getElementById('play-icon').classList.add('hidden');
        document.getElementById('pause-icon').classList.remove('hidden');
        
        updateStatus('Detecting vehicles in video...', 'loading');
        
        // Start processing frames
        requestAnimationFrame(videoFrameLoop);
    }
}

function stopVideoLoop() {
    isVideoPlaying = false;
    document.getElementById('play-icon').classList.remove('hidden');
    document.getElementById('pause-icon').classList.add('hidden');
    
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
    
    updateStatus('Video paused', 'active');
}

// Core loop processing video frames
async function videoFrameLoop() {
    if (hiddenVideo.paused || hiddenVideo.ended || !isLoaded || !isVideoPlaying) {
        return;
    }

    if (!isProcessingFrame) {
        isProcessingFrame = true;
        
        const startTime = performance.now();
        const results = await runInference(hiddenVideo);
        const duration = Math.round(performance.now() - startTime);
        
        if (results && isVideoPlaying) {
            renderDetections(hiddenVideo, results, duration);
        }
        
        isProcessingFrame = false;
    }

    if (isVideoPlaying) {
        animationFrameId = requestAnimationFrame(videoFrameLoop);
    }
}

// Video bar updates
function updateVideoProgress() {
    if (!hiddenVideo.duration) return;
    const percentage = (hiddenVideo.currentTime / hiddenVideo.duration) * 100;
    videoProgress.style.width = `${percentage}%`;
    updateVideoTime(hiddenVideo.currentTime, hiddenVideo.duration);
}

function updateVideoTime(current, total) {
    const format = (time) => {
        const mins = Math.floor(time / 60);
        const secs = Math.floor(time % 60);
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    };
    videoTime.textContent = `${format(current)} / ${format(total)}`;
}

// Webcam integration functions
let isWebcamActive = false;

async function toggleWebcam() {
    if (!isLoaded) {
        alert("Please initialize the model before starting webcam detection.");
        return;
    }

    if (isWebcamActive) {
        // Stop camera stream
        cleanupActiveModes();
        isWebcamActive = false;
        btnWebcamToggle.textContent = "Start Webcam";
        btnWebcamToggle.className = "btn btn-primary mt-4";
        
        // Hide canvas, show placeholder
        canvasContainer.classList.add('hidden');
        webcamPlaceholder.classList.remove('hidden');
        
        resetResults();
        updateStatus('Webcam feed stopped', 'idle');
    } else {
        // Start camera stream
        updateStatus('Starting webcam...', 'loading');
        try {
            webcamStream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: 'environment', width: 640, height: 480 },
                audio: false
            });
            
            hiddenVideo.onloadedmetadata = () => {
                outputCanvas.width = hiddenVideo.videoWidth;
                outputCanvas.height = hiddenVideo.videoHeight;
                
                updateStatus('Live webcam detection active', 'active');
                
                // Start processing loop
                requestAnimationFrame(webcamFrameLoop);
            };

            hiddenVideo.srcObject = webcamStream;
            hiddenVideo.play();
            
            isWebcamActive = true;
            btnWebcamToggle.textContent = "Stop Webcam";
            btnWebcamToggle.className = "btn btn-secondary mt-4";
            
            // Hide placeholder, prepare canvas
            webcamPlaceholder.classList.add('hidden');
            canvasContainer.classList.remove('hidden');
            
            // Show action buttons (only Clear Workspace will be visible in webcam mode)
            updateActionsBar();
            
        } catch (err) {
            console.error("Camera access failed:", err);
            alert("Could not access camera. Please check permissions and ensure no other application is using it.");
            updateStatus('Camera connection failed', 'idle');
        }
    }
}

// Core loop processing webcam frames
async function webcamFrameLoop() {
    if (!isWebcamActive || !isLoaded) return;

    if (!isProcessingFrame) {
        isProcessingFrame = true;
        
        const startTime = performance.now();
        const results = await runInference(hiddenVideo);
        const duration = Math.round(performance.now() - startTime);
        
        if (results && isWebcamActive) {
            renderDetections(hiddenVideo, results, duration);
        }
        
        isProcessingFrame = false;
    }

    if (isWebcamActive) {
        animationFrameId = requestAnimationFrame(webcamFrameLoop);
    }
}

// Action Bar & Workspace Control Functions
function updateActionsBar() {
    if (!workspaceActions) return;
    
    // Show actions bar if canvas is active
    if (!canvasContainer.classList.contains('hidden')) {
        workspaceActions.classList.remove('hidden');
        
        // In image mode, show both Submit and Clear
        if (currentMode === 'image') {
            btnSubmit.classList.remove('hidden');
        } else {
            // In video/webcam mode, only show Clear (as play/stop act as submit)
            btnSubmit.classList.add('hidden');
        }
    } else {
        workspaceActions.classList.add('hidden');
    }
}

function clearWorkspace() {
    cleanupActiveModes();
    resetResults();
    
    // Hide canvas and actions
    canvasContainer.classList.add('hidden');
    if (workspaceActions) workspaceActions.classList.add('hidden');
    videoControls.classList.add('hidden');
    
    // Show original drop-zone/placeholder
    const workspacePanel = document.getElementById(`workspace-${currentMode}`);
    if (workspacePanel) {
        workspacePanel.classList.remove('hidden');
    }
    
    // Reset file inputs
    imageFileInput.value = '';
    videoFileInput.value = '';
    
    // Clear canvas drawing
    const ctx = outputCanvas.getContext('2d');
    ctx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
    
    // Remove loaded image elements
    const oldImg = document.getElementById('loaded-image');
    if (oldImg) oldImg.remove();
    
    updateStatus('Workspace cleared', 'idle');
}

async function submitImageDetection() {
    const img = document.getElementById('loaded-image');
    if (!img) {
        alert("No image loaded in workspace. Please drag & drop or browse an image file.");
        return;
    }
    if (!isLoaded) {
        alert("Model is still initializing. Please wait.");
        return;
    }
    
    inferenceLoading.classList.remove('hidden');
    updateStatus('Running vehicle detection...', 'loading');
    
    // Let rendering finish before heavy CPU inference
    setTimeout(async () => {
        try {
            const startTime = performance.now();
            const results = await runInference(img);
            const duration = Math.round(performance.now() - startTime);
            
            if (results) {
                renderDetections(img, results, duration);
                updateStatus('Detection completed', 'active');
            }
        } catch (err) {
            console.error("Image detection error:", err);
            updateStatus('Detection failed', 'idle');
        } finally {
            inferenceLoading.classList.add('hidden');
        }
    }, 50);
}
