document.addEventListener('DOMContentLoaded', function () {
    const formContainer = document.getElementById('form-container');
    const form = document.createElement('form');
    form.setAttribute('method', 'post');
    form.setAttribute('action', '/update-circle-params');

    const sliders = [
        {name: 'zoom', path: "/form_update", min: 1, max: 5, step: 0.1, defaultValue: 1, update_endpoints: true},
        {name: 'resize', path: "/form_update", min: 0, max: 100, step: 1, defaultValue: 50, update_endpoints: false},
        {name: 'min-area', path: "/form_update", min: 5000, max: 1000000, step: 1, defaultValue: 5000, update_endpoints: true},
        {name: 'minDist', path: "/form_update", min: 0, max: 100, step: 1, defaultValue: 50, update_endpoints: true},
        {name: 'max-area', path: "/form_update", min: 0, max: 100, step: 1, defaultValue: 50, update_endpoints: true},
        {name: 'param2', path: "/form_update", min: 0, max: 100, step: 1, defaultValue: 50, update_endpoints: true},
        {name: 'minRadius', path: "/form_update", min: 0, max: 100, step: 1, defaultValue: 50, update_endpoints: true},
        {name: 'maxRadius', path: "/form_update", min: 0, max: 100, step: 1, defaultValue: 50, update_endpoints: true}
    ];

    sliders.forEach(sliderData => {
        const slider = document.createElement('input');
        slider.type = 'range';
        slider.name = sliderData.name;
        slider.min = sliderData.min;
        slider.max = sliderData.max;
        slider.step = sliderData.step;
        slider.value = sliderData.defaultValue;
        slider.onchange = function () {
            autoSubmitSliderValue(slider.name, slider.value);
        };
        form.appendChild(slider);
        form.appendChild(document.createElement('br')); // For better spacing
    });

    formContainer.appendChild(form);

    function updateSliderValue(name, value) {
        console.log(`${name} Slider Value:`, value);
    }

    function autoSubmitSliderValue(name, value) {
        console.log(`${name} Slider Value:`, value); // Debugging output
        // Automatically submit slider value
        const formData = {[name]: value};
        fetch(form.action, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        })
            .then(response => response.json())
            .then(data => console.log(`${name} Updated:`, data))
            .catch(error => console.error('Error:', error));
    }
});