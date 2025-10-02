document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const previewContainer = document.getElementById('previewContainer');
    const imagePreview = document.getElementById('imagePreview');
    const classifyBtn = document.getElementById('classifyBtn');
    const resultContainer = document.getElementById('resultContainer');
    const resetBtn = document.getElementById('resetBtn');
    const loader = document.getElementById('loader');
    const confidenceBar = document.getElementById('confidenceBar');
    const confidenceText = document.getElementById('confidence');
    const fruitClass = document.getElementById('fruitClass');
    const fruitIcon = document.getElementById('fruitIcon');
    const timestamp = document.getElementById('timestamp');
    
    // Grad-CAM elements
    const gradcamContainer = document.getElementById('gradcamContainer');
    const viewGradcamBtn = document.getElementById('viewGradcamBtn');
    const backToResultsBtn = document.getElementById('backToResults');
    const downloadGradcamBtn = document.getElementById('downloadGradcam');
    const gradcamTabs = document.querySelectorAll('.gradcam-tab');
    const gradcamTabContents = document.querySelectorAll('.gradcam-tab-content');
    const originalImage = document.getElementById('originalImage');
    const heatmapImage = document.getElementById('heatmapImage');
    const overlayImage = document.getElementById('overlayImage');
    
    // Export report button
    const exportReportBtn = document.querySelector('.action-buttons .cyber-btn:last-child');
    
    // Debug elements (optional)
    const debugPanel = document.querySelector('.debug-panel');
    const debugLogs = document.getElementById('debugLogs');
    
    // Current selected file and classification results
    let currentFile = null;
    let classificationResult = null;
    let gradcamResult = null;
    let currentFruitId = null; // Added for database integration
    
    // Debug function (optional)
    function debugLog(message, type = 'log') {
        if (!debugPanel) return;
        
        debugPanel.style.display = 'block';
        const logEntry = document.createElement('div');
        logEntry.className = `log ${type}`;
        logEntry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
        debugLogs.appendChild(logEntry);
        debugLogs.scrollTop = debugLogs.scrollHeight;
        
        // Keep only last 10 logs
        while (debugLogs.children.length > 10) {
            debugLogs.removeChild(debugLogs.firstChild);
        }
    }
    
    debugLog('System initialized', 'success');
    
    // File selection handler
    fileInput.addEventListener('change', handleFileSelect);
    
    // Make sure the label triggers the file input
    document.querySelector('label[for="fileInput"]').addEventListener('click', (e) => {
        e.preventDefault();
        fileInput.click();
    });
    
    // Drag and drop handlers
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
        debugLog('File dragged over drop zone');
    });
    
    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
        debugLog('File left drop zone');
    });
    
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        
        if (e.dataTransfer.files.length) {
            fileInput.files = e.dataTransfer.files;
            debugLog(`Dropped file: ${e.dataTransfer.files[0].name}`, 'success');
            handleFileSelect();
        }
    });
    
    // Classify button handler
    classifyBtn.addEventListener('click', classifyImage);
    
    // Reset button handler
    resetBtn.addEventListener('click', resetApp);
    
    // View Grad-CAM button handler
    viewGradcamBtn.addEventListener('click', async () => {
        if (!currentFruitId) {
            debugLog('No fruit ID available for Grad-CAM', 'error');
            return;
        }
        
        debugLog('Loading Grad-CAM visualization...', 'success');
        
        // Show loader
        resultContainer.style.display = 'none';
        loader.style.display = 'flex';
        
        try {
            // Create FormData for fruit_id
            const formData = new FormData();
            formData.append('fruit_id', currentFruitId);
            
            debugLog('Sending request to /gradcam endpoint...');
            
            // Make API call to backend
            const response = await fetch('/gradcam', {
                method: 'POST',
                body: formData
            });
            
            debugLog(`Response status: ${response.status}`);
            
            if (!response.ok) {
                const errorData = await response.json();
                
                // Handle non-fruit classification error
                if (errorData.error && errorData.error.includes('not classified as a fruit')) {
                    debugLog('Grad-CAM failed: Image not classified as a fruit', 'warning');
                    // Hide loader
                    loader.style.display = 'none';
                    // Show error message
                    alert('This image is not classified as a fruit. Grad-CAM visualization is not available.');
                    // Reset to results
                    resultContainer.style.display = 'block';
                    return;
                }
                
                throw new Error(errorData.error || `HTTP ${response.status}`);
            }
            
            const result = await response.json();
            debugLog(`Grad-CAM generation successful`, 'success');
            
            // Check if we have valid images
            if (!result.original || !result.heatmap || !result.overlay) {
                throw new Error('Invalid Grad-CAM response');
            }
            
            // Store Grad-CAM result
            gradcamResult = result;
            
            // Set images
            originalImage.src = result.original;
            heatmapImage.src = result.heatmap;
            overlayImage.src = result.overlay;
            
            // Hide loader
            loader.style.display = 'none';
            
            // Show Grad-CAM section
            gradcamContainer.style.display = 'block';
            
            // Set first tab as active
            gradcamTabs[0].click();
            
        } catch (error) {
            debugLog(`Grad-CAM error: ${error.message}`, 'error');
            
            // Hide loader
            loader.style.display = 'none';
            
            // Show error message
            alert(`Grad-CAM generation failed: ${error.message}`);
            
            // Reset to results
            resultContainer.style.display = 'block';
        }
    });
    
    // Back to results button handler
    backToResultsBtn.addEventListener('click', () => {
        gradcamContainer.style.display = 'none';
        resultContainer.style.display = 'block';
    });
    
    // Grad-CAM tab handlers
    gradcamTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            // Remove active class from all tabs and contents
            gradcamTabs.forEach(t => t.classList.remove('active'));
            gradcamTabContents.forEach(c => c.classList.remove('active'));
            
            // Add active class to clicked tab
            tab.classList.add('active');
            
            // Show corresponding content
            const tabId = tab.getAttribute('data-tab');
            document.getElementById(`${tabId}-tab`).classList.add('active');
        });
    });
    
    // Download Grad-CAM button handler
    downloadGradcamBtn.addEventListener('click', () => {
        // Get active tab
        const activeTab = document.querySelector('.gradcam-tab.active');
        const tabId = activeTab.getAttribute('data-tab');
        
        // Get corresponding image
        const img = document.getElementById(`${tabId}Image`);
        
        // Create download link
        const link = document.createElement('a');
        link.download = `gradcam-${tabId}-${Date.now()}.jpg`;
        link.href = img.src;
        link.click();
        
        debugLog(`Downloaded Grad-CAM ${tabId} image`, 'success');
    });
    
    // Export report button handler
    exportReportBtn.addEventListener('click', async () => {
        if (!classificationResult) {
            alert('No classification results available. Please classify an image first.');
            return;
        }
        
        // Check if the image was classified as a fruit
        if (classificationResult.error && classificationResult.error.includes('not classified as a fruit')) {
            alert('This image is not classified as a fruit. Report generation is not available.');
            return;
        }
        
        debugLog('Generating report...', 'success');
        
        try {
            // Prepare data for report
            const reportData = {
                original: imagePreview.src,
                heatmap: gradcamResult ? gradcamResult.original : null,
                overlay: gradcamResult ? gradcamResult.overlay : null,
                prediction: classificationResult.class,
                confidence: classificationResult.confidence,
                top_predictions: classificationResult.top_predictions,
                fruit_id: currentFruitId
            };
            
            debugLog('Sending request to /generate_report endpoint...');
            
            // Make API call to backend
            const response = await fetch('/generate_report', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(reportData)
            });
            
            debugLog(`Response status: ${response.status}`);
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP ${response.status}`);
            }
            
            // Get the blob from response
            const blob = await response.blob();
            
            // Create download link
            const url = window.URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = `fruit_classification_report_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.pdf`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            window.URL.revokeObjectURL(url);
            
            debugLog('Report downloaded successfully', 'success');
            
        } catch (error) {
            debugLog(`Report generation error: ${error.message}`, 'error');
            alert(`Failed to generate report: ${error.message}`);
        }
    });
    
    function handleFileSelect() {
        debugLog('handleFileSelect called');
        
        if (fileInput.files.length === 0) {
            debugLog('No files selected', 'error');
            return;
        }
        
        const file = fileInput.files[0];
        currentFile = file;
        debugLog(`Selected file: ${file.name} (${file.type}, ${file.size} bytes)`);
        
        // Check if file is an image
        if (!file.type.startsWith('image/')) {
            debugLog('File is not an image', 'error');
            alert('Please select an image file');
            return;
        }
        
        // Check file size (16MB limit)
        if (file.size > 16 * 1024 * 1024) {
            debugLog('File too large', 'error');
            alert('File size must be less than 16MB');
            return;
        }
        
        const reader = new FileReader();
        
        reader.onload = (e) => {
            debugLog('FileReader loaded successfully', 'success');
            imagePreview.src = e.target.result;
            dropZone.style.display = 'none';
            previewContainer.style.display = 'flex';
            debugLog('Preview container shown', 'success');
        };
        
        reader.onerror = (error) => {
            debugLog(`FileReader error: ${error}`, 'error');
        };
        
        reader.readAsDataURL(file);
        debugLog('Reading file as Data URL...');
    }
    
    async function classifyImage() {
        if (!currentFile) {
            debugLog('No file selected for classification', 'error');
            return;
        }
        
        debugLog('Starting real classification...', 'success');
        
        // Show loader
        previewContainer.style.display = 'none';
        loader.style.display = 'flex';
        
        try {
            // Create FormData for file upload
            const formData = new FormData();
            formData.append('image', currentFile);
            
            debugLog('Sending request to /classify endpoint...');
            
            // Make API call to backend
            const response = await fetch('/classify', {
                method: 'POST',
                body: formData
            });
            
            debugLog(`Response status: ${response.status}`);
            
            const result = await response.json();
            
            // Check if response contains an error
            if (!response.ok) {
                // Handle non-fruit classification error
                if (result.error && result.error.includes('not classified as a fruit')) {
                    debugLog(`Image not classified as a fruit. Confidence: ${result.confidence}`, 'warning');
                    
                    // Store the result for potential report generation
                    classificationResult = result;
                    
                    // Hide loader
                    loader.style.display = 'none';
                    
                    // Show non-fruit result
                    showNonFruitResult(result);
                    return;
                }
                
                throw new Error(result.error || `HTTP ${response.status}`);
            }
            
            debugLog(`Classification successful: ${JSON.stringify(result)}`, 'success');
            
            // Store fruit ID from response
            currentFruitId = result.fruit_id;
            
            // Store classification result
            classificationResult = result;
            
            // Hide loader
            loader.style.display = 'none';
            
            // Show results
            showResults(result);
            
        } catch (error) {
            debugLog(`Classification error: ${error.message}`, 'error');
            
            // Hide loader
            loader.style.display = 'none';
            
            // Show error message
            alert(`Classification failed: ${error.message}`);
            
            // Reset to preview
            previewContainer.style.display = 'flex';
        }
    }
    
    function showResults(result) {
        // Set timestamp
        timestamp.textContent = result.timestamp || new Date().toLocaleTimeString();
        
        // Set fruit class with typing effect
        fruitClass.textContent = '';
        typeWriter(fruitClass, result.class, 100);
        
        // Set fruit icon (use the icon from backend or fallback)
        const iconClass = result.icon || 'fa-question';
        fruitIcon.className = `fas ${iconClass}`;
        
        // Animate confidence bar
        setTimeout(() => {
            const confidencePercent = (result.confidence * 100);
            confidenceBar.style.width = `${confidencePercent}%`;
            confidenceText.textContent = `${confidencePercent.toFixed(1)}%`;
            
            // Color coding based on confidence
            if (confidencePercent >= 90) {
                confidenceBar.style.background = 'linear-gradient(90deg, #00ff88, #00cc6a)';
            } else if (confidencePercent >= 70) {
                confidenceBar.style.background = 'linear-gradient(90deg, #ffaa00, #ff8800)';
            } else {
                confidenceBar.style.background = 'linear-gradient(90deg, #ff4444, #cc3333)';
            }
        }, 1000);
        
        // Show results
        resultContainer.style.display = 'block';
        
        // Animate neural network
        animateNeuralNetwork();
        
        debugLog(`Results displayed: ${result.class} (${(result.confidence * 100).toFixed(1)}%)`, 'success');
        
        // Log top predictions if available
        if (result.top_predictions) {
            debugLog(`Top predictions: ${result.top_predictions.map(p => `${p.class}: ${(p.confidence * 100).toFixed(1)}%`).join(', ')}`);
        }
    }
    
    function showNonFruitResult(result) {
        // Set timestamp
        timestamp.textContent = result.timestamp || new Date().toLocaleTimeString();
        
        // Update UI to show non-fruit result
        fruitClass.textContent = 'Not a Fruit';
        fruitIcon.className = 'fas fa-times-circle';
        
        // Animate confidence bar
        setTimeout(() => {
            const confidencePercent = (result.confidence * 100);
            confidenceBar.style.width = `${confidencePercent}%`;
            confidenceText.textContent = `${confidencePercent.toFixed(1)}%`;
            
            // Always use red for non-fruit classification
            confidenceBar.style.background = 'linear-gradient(90deg, #ff4444, #cc3333)';
        }, 1000);
        
        // Show results
        resultContainer.style.display = 'block';
        
        // Disable Grad-CAM and report buttons for non-fruit images
        viewGradcamBtn.disabled = true;
        viewGradcamBtn.classList.add('disabled');
        exportReportBtn.disabled = true;
        exportReportBtn.classList.add('disabled');
        
        // Create a non-fruit result message
        const nonFruitMessage = document.createElement('div');
        nonFruitMessage.className = 'non-fruit-result';
        nonFruitMessage.innerHTML = `
            <h3>Not a Fruit</h3>
            <p>This image doesn't appear to be a fruit. Please upload an image of a fruit.</p>
            <div class="low-confidence-indicator">
                <i class="fas fa-exclamation-triangle"></i>
                <span>Model confidence: ${(result.confidence * 100).toFixed(1)}%</span>
            </div>
        `;
        
        // Add to result container
        resultContainer.appendChild(nonFruitMessage);
        
        // Show top predictions if available
        if (result.top_predictions && result.top_predictions.length > 0) {
            // Create a container for top predictions
            const topPredictionsContainer = document.createElement('div');
            topPredictionsContainer.className = 'top-predictions';
            topPredictionsContainer.innerHTML = '<h5>Top Predictions:</h5><ul>';
            
            result.top_predictions.forEach(pred => {
                topPredictionsContainer.innerHTML += `<li><span class="prediction-class">${pred.class}</span><span class="prediction-confidence">${(pred.confidence * 100).toFixed(1)}%</span></li>`;
            });
            
            topPredictionsContainer.innerHTML += '</ul>';
            
            // Add to result container
            resultContainer.appendChild(topPredictionsContainer);
        }
        
        debugLog(`Non-fruit result displayed. Confidence: ${(result.confidence * 100).toFixed(1)}%`, 'warning');
    }
    
    function typeWriter(element, text, speed) {
        let i = 0;
        function type() {
            if (i < text.length) {
                element.textContent += text.charAt(i);
                i++;
                setTimeout(type, speed);
            }
        }
        type();
    }
    
    function animateNeuralNetwork() {
        const nodes = document.querySelectorAll('.network-nodes .node');
        nodes.forEach((node, index) => {
            setTimeout(() => {
                node.classList.add('active');
            }, index * 200);
        });
    }
    
    function resetApp() {
        debugLog('Resetting app', 'success');
        
        dropZone.style.display = 'flex';
        previewContainer.style.display = 'none';
        resultContainer.style.display = 'none';
        gradcamContainer.style.display = 'none';
        loader.style.display = 'none';
        fileInput.value = '';
        currentFile = null;
        classificationResult = null;
        gradcamResult = null;
        currentFruitId = null;
        confidenceBar.style.width = '0%';
        confidenceText.textContent = '0%';
        fruitClass.textContent = '';
        
        // Reset confidence bar color
        confidenceBar.style.background = '';
        
        // Reset neural network animation
        document.querySelectorAll('.network-nodes .node').forEach(node => {
            node.classList.remove('active');
        });
        
        // Re-enable buttons
        viewGradcamBtn.disabled = false;
        viewGradcamBtn.classList.remove('disabled');
        exportReportBtn.disabled = false;
        exportReportBtn.classList.remove('disabled');
        
        // Remove non-fruit result message if it exists
        const nonFruitMessage = document.querySelector('.non-fruit-result');
        if (nonFruitMessage) {
            nonFruitMessage.remove();
        }
        
        // Remove top predictions container if it exists
        const topPredictionsContainer = document.querySelector('.top-predictions');
        if (topPredictionsContainer) {
            topPredictionsContainer.remove();
        }
    }
    
    // Health check on load
    async function checkServerHealth() {
        try {
            const response = await fetch('/health');
            const health = await response.json();
            debugLog(`Server health: ${health.status}, Classes: ${health.classes_loaded}`, 'success');
        } catch (error) {
            debugLog(`Health check failed: ${error.message}`, 'error');
        }
    }
    
    // Check server health on load
    checkServerHealth();
});