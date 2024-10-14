let wheels = document.getElementsByClassName('jogWheel');
let centerDots = document.getElementsByClassName('centerDot');
let lastAngle = 0;
let accumulatedDelta = 0;
let smoothedDelta = 0;
let lastTime = 0;
const throttleDelay = 100;
let maxPerRotation = 30; // Max delta for one full rotation
let angleToDeltaRatio = 360 / maxPerRotation; // 360 degrees corresponds to max delta (30)
const smoothingFactor = 0.2; // Lower value means more smoothing


function getAngle(x, y, cx, cy) {
    let dy = y - cy;
    let dx = x - cx;
    let theta = Math.atan2(dy, dx); // range (-PI, PI]
    theta *= 180 / Math.PI; // Radians to degrees
    return theta;
}


function applySmoothing(currentDelta) {
    // Apply a low-pass filter to smooth the delta value
    smoothedDelta = smoothedDelta + smoothingFactor * (currentDelta - smoothedDelta);
    return smoothedDelta;
}


[...centerDots].forEach(function (centerDot) {
    centerDot.addEventListener('click', function (event) {
        // Function to toggle between 30 per revolution and 1 per revolution
        if (maxPerRotation === 30) {
            maxPerRotation = 1; // Switch to 1 per revolution
            centerDot.classList.add('micro_adjust'); // Change color to indicate micro_adjust state
        } else {
            maxPerRotation = 30; // Switch back to 30 per revolution
            centerDot.classList.remove('micro_adjust');
        }
    });

});

[...wheels].forEach(function (wheel) {

    wheel.addEventListener('mousedown', function (event) {
        const rect = wheel.getBoundingClientRect();
        const cx = rect.left + rect.width / 2;
        const cy = rect.top + rect.height / 2;

        function updateValueDisplay() {
            valueDisplay.textContent = slider.value;
        }

        function onMouseMove(e) {
            const angle = getAngle(e.clientX, e.clientY, cx, cy);
            let delta = (angle - lastAngle) / angleToDeltaRatio; // Normalize the delta
            const dot = wheel.querySelector('.centerDot');
            if (dot.classList.contains('micro_adjust')) {
                console.log('Dot is micro_adjust');
                maxPerRotation = 1
            } else {
                maxPerRotation = 30
                console.log('Dot is not micro_adjust');
            }
            angleToDeltaRatio = 360 / maxPerRotation; // Update ratio based on new maxPerRotation
            // Handle crossing the 360-degree threshold
            if (delta > 180 / angleToDeltaRatio) delta -= 360 / angleToDeltaRatio;
            if (delta < -180 / angleToDeltaRatio) delta += 360 / angleToDeltaRatio;

            // Accumulate delta and apply smoothing
            accumulatedDelta += delta;
            let smoothed = applySmoothing(accumulatedDelta);

            lastAngle = angle;

            let now = Date.now();
            if (now - lastTime > throttleDelay) {
                sendJogData(smoothed, wheel.id);
                accumulatedDelta = 0;
                lastTime = now;
            }
        }

        function onMouseUp() {
            document.removeEventListener('mousemove', onMouseMove);
            document.removeEventListener('mouseup', onMouseUp);
        }

        document.addEventListener('mousemove', onMouseMove);
        document.addEventListener('mouseup', onMouseUp);

        lastAngle = getAngle(event.clientX, event.clientY, cx, cy);
    });

    function sendJogData(delta, jogAxis) {
        fetch('/jog', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({delta: delta, axis: jogAxis})
        })
            .then(response => response.json())
            .then(data => {
                console.log('Success:', data);
                // xvalue_elem = document.getElementById('x-value');
                // xvalue_elem.value = data['coords'][0];
                // yvalue_elem = document.getElementById('y-value');
                // yvalue_elem.value = data['coords'][1];
                // zvalue_elem = document.getElementById('z-value');
                // zvalue_elem.value = data['coords'][2];
            })
            .catch((error) => {
                console.error('Error:', error);
            });
    };
});