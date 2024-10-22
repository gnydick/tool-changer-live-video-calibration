document.addEventListener('DOMContentLoaded', (event) => {
        
        function applyZoom(value) {
            return value * zoom_level
        }


        function jog(direction) {
            // Implement the logic to handle jog control
            console.log('Jogging', direction);
        }


        function setupParameterSlider(minInputId, sliderId, maxInputId, valueId) {
            var minInput = document.getElementById(minInputId);
            var slider = document.getElementById(sliderId);
            var maxInput = document.getElementById(maxInputId);
            var valueDisplay = document.getElementById(valueId);

            function updateValueDisplay() {
                valueDisplay.textContent = slider.value;
            }

            function updateSliderLimits() {
                slider.min = minInput.value;
                slider.max = maxInput.value;
                slider.value = Math.min(slider.value, slider.max);
                updateValueDisplay();
            }


            minInput.addEventListener('change', updateSliderLimits);
            maxInput.addEventListener('change', updateSliderLimits);
            slider.addEventListener('input', updateValueDisplay);

            updateSliderLimits()

            function sendSliderData() {
                var formData = {
                    dp: document.getElementById('dp-slider').value,
                    zoom: document.getElementById('zoom-slider').value,
                    center_precision: document.getElementById('center_precision-slider').value,
                    reticle_rad: document.getElementById('reticle_rad-slider').value,
                    minDist: document.getElementById('minDist-slider').value,
                    param1: document.getElementById('param1-slider').value,
                    param2: document.getElementById('param2-slider').value,
                    minRadius: document.getElementById('minRadius-slider').value,
                    maxRadius: document.getElementById('maxRadius-slider').value,
                    // tool: getSelectedTool('dynamic-tools').value
                };

                fetch('/update-circle-params', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(formData)
                })
                    .then(response => response.json())
                    .then(data => {
                        console.log('Success:', data);
                    })
                    .catch((error) => {
                        console.error('Error:', error);
                    });
            }

            const events = ['mouseup', 'touchend'];

            function handleSliderChange() {
                updateValueDisplay();
                sendSliderData(); // Send data when the slider value changes
            }

            events.forEach(event => {
                slider.addEventListener(event, handleSliderChange);
            });


            minInput.onchange = sendSliderData;
            maxInput.onchange = sendSliderData;

            updateValueDisplay(); // Initialize at setup
        }

// Initialize sliders for each parameter
        setupParameterSlider('center_precision-min', 'center_precision-slider', 'center_precision-max', 'center_precision-value');
        setupParameterSlider('reticle_rad-min', 'reticle_rad-slider', 'reticle_rad-max', 'reticle_rad-value');
        setupParameterSlider('zoom-min', 'zoom-slider', 'zoom-max', 'zoom-value');
        setupParameterSlider('dp-min', 'dp-slider', 'dp-max', 'dp-value');
        setupParameterSlider('minDist-min', 'minDist-slider', 'minDist-max', 'minDist-value');
        setupParameterSlider('param1-min', 'param1-slider', 'param1-max', 'param1-value');
        setupParameterSlider('param2-min', 'param2-slider', 'param2-max', 'param2-value');
        setupParameterSlider('minRadius-min', 'minRadius-slider', 'minRadius-max', 'minRadius-value');
        setupParameterSlider('maxRadius-min', 'maxRadius-slider', 'maxRadius-max', 'maxRadius-value');

        function getSelectedTool(containerId) {
            var container = document.getElementById(containerId);
            return container.querySelector('.selected');
        }

        document.getElementById('form-container').addEventListener('change', function (event) {
            var formData = {
                dp: document.getElementById('dp-slider').value,
                center_precision: document.getElementById('center_precision-slider').value,
                reticle_rad: document.getElementById('reticle_rad-slider').value,
                zoom: document.getElementById('zoom-slider').value,
                minDist: document.getElementById('minDist-slider').value,
                param1: document.getElementById('param1-slider').value,
                param2: document.getElementById('param2-slider').value,
                minRadius: document.getElementById('minRadius-slider').value,
                maxRadius: document.getElementById('maxRadius-slider').value,
                // tool: getSelectedTool('dynamic-tools').value
            };

            fetch('/update-circle-params', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            })
                .then(response => response.json())
                .then(data => {
                    console.log('Success:', data);
                })
                .catch((error) => {
                    console.error('Error:', error);
                });
        });


        //  selection box instead of buttons
        // let isDragging = false;
        // let isResizing = false;
        // let dragStartX, dragStartY;
        // const selectionSquare = document.getElementById('selection-square');
        // const resizeHandle = document.getElementById('resize-handle');
        // const cameraFeed = document.getElementById('camera-feed');
        // const image = document.getElementById('camera-feed-img');
        //
        //
        //
        // selectionSquare.addEventListener('mousedown', function(e) {
        //     if (e.target === resizeHandle) {
        //         isResizing = true;
        //         e.preventDefault(); // Prevent default drag behavior when resizing
        //     } else {
        //         isDragging = true;
        //         dragStartX = e.clientX - selectionSquare.offsetLeft;
        //         dragStartY = e.clientY - selectionSquare.offsetTop;
        //     }
        //     cameraFeed.addEventListener('mousemove', onMouseMove);
        //     cameraFeed.addEventListener('mousemove', onResize);
        // });
        //
        // document.addEventListener('mouseup', function() {
        //     isDragging = false;
        //     isResizing = false;
        //     cameraFeed.removeEventListener('mousemove', onMouseMove);
        //     cameraFeed.removeEventListener('mousemove', onResize);
        // });
        //
        // function onMouseMove(e) {
        //     if (!isDragging) return;
        //     const rect = image.getBoundingClientRect();
        //     const maxX = rect.width + rect.left - selectionSquare.offsetWidth;
        //     const maxY = rect.height + rect.top - selectionSquare.offsetHeight;
        //
        //     let x = e.clientX - dragStartX;
        //     let y = e.clientY - dragStartY;
        //
        //     x = Math.min(Math.max(rect.left, x), maxX) - rect.left;
        //     y = Math.min(Math.max(rect.top, y), maxY) - rect.top;
        //
        //     selectionSquare.style.left = `${x}px`;
        //     selectionSquare.style.top = `${y}px`;
        // }
        //
        // function onResize(e) {
        //     if (!isResizing) return;
        //     const rect = image.getBoundingClientRect();
        //     const maxX = e.clientX - selectionSquare.offsetLeft;
        //     const maxY = e.clientY - selectionSquare.offsetTop;
        //
        //     let width = Math.min(maxX, rect.width);
        //     let height = Math.min(maxY, rect.height);
        //
        //     selectionSquare.style.width = `${width}px`;
        //     selectionSquare.style.height = `${height}px`;
        // }


    }
)
;


function pan(direction) {
    const panDistance = document.getElementById('pan-distance').value;
    var formData = {
        direction: direction,
        pan_distance: panDistance
    };
    // Example: Send the pan direction and distance to the server
    fetch('/pan', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData)
    })
        .then(response => response.json())
        .then(data => {
            console.log('Success:', data);
        })
        .catch((error) => {
            console.error('Error:', error);
        });
}

function resize() {
    const sizeLevel = document.getElementById('size-slider').value;
    var valueDisplay = document.getElementById("size-value");
    valueDisplay.textContent = sizeLevel + '%';
    resizeVideo()
}

function reticle_rad() {
    const reticle_radSlider = document.getElementById('reticle_rad-slider');
    var reticle_radLevel = reticle_radSlider.value;
    var valueDisplay = document.getElementById("reticle_rad-value");

    valueDisplay.textContent = reticle_radLevel;
    reticle_radSlider.value = reticle_radLevel
    console.log('Success:', data);
}

function center_precision() {
    const center_precisionSlider = document.getElementById('center_precision-slider');
    var center_precisionLevel = center_precisionSlider.value;
    var valueDisplay = document.getElementById("center_precision-value");

    valueDisplay.textContent = center_precisionLevel;
    center_precisionSlider.value = center_precisionLevel
    console.log('Success:', data);
}

function zoom() {
    const zoomSlider = document.getElementById('zoom-slider');
    var zoomLevel = zoomSlider.value;
    var valueDisplay = document.getElementById("zoom-value");

        valueDisplay.textContent = zoomLevel + 'x';
        zoomSlider.value = zoomLevel
        console.log('Success:', data);
}

function selectTool(index) {
    const tools = document.getElementById('dynamic-tools').children;
    for (let i = 0; i < tools.length; i++) {
        if (i === index) {
            tools[i].classList.add('selected');
        } else {
            tools[i].classList.remove('selected');
        }
    }
}

function createTools() {
    var selected = document.querySelectorAll('button.tool.selected');
    const count = Math.max(1, document.getElementById('toolCount').value); // Ensure at least 1
    const container = document.getElementById('dynamic-tools');
    container.innerHTML = ''; // Clear existing buttons


    for (let i = 0; i < count; i++) {
        const button = document.createElement('button');
        button.id = i;
        button.className = "tool";
        if (selected.length === 1) {
            if (selected[0].id === i.toString()) {
                button.classList.add('selected');
            }
        }
        if (count === 1) {
            button.classList.add('selected');
        }
        button.setAttribute("value", i.toString())
        button.innerText = `Tool ${i}`;
        button.onclick = function () {
            selectTool(i);
        };
        container.appendChild(button);
    }
}

function initialResize() {
    document.getElementById('camera-feed').style.width = '100%';
    createTools();
}

function resizeVideo() {
    document.getElementById('size-slider').addEventListener('input', function () {
        var video = document.getElementById('camera-feed'); // Ensure this id matches your video element
        var w = video.style.width;
        var sliderValue = this.value;
        video.style.width = sliderValue + '%';
        video.style.height = 'auto';
        this.textContent = sliderValue + '%';

    });

}

function lock() {

    // Example: Send the pan direction and distance to the server
    fetch('/toggle-lock', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }

    }).then(response => response.json()).then(data => {
        var button = document.getElementById('lockButton')
        button.value = data['locked_state'];
        console.log('Success:', data);
    }).catch((error) => {
        console.error('Error:', error);
    });
}
