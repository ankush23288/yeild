document.addEventListener('DOMContentLoaded', function() {
    const predictBtn = document.getElementById('predict-btn');
    const cropSelect = document.getElementById('crop');
    const featureInputsDiv = document.getElementById('feature-inputs');
    const predictionOutputDiv = document.getElementById('prediction-output');
    const shapPlotDiv = document.getElementById('shap-plot');
    const recommendationsUl = document.querySelector('#recommendations ul');

    // Function to update slider value display
    function updateSliderValue(event) {
        const valueSpan = document.getElementById(event.target.id + '-value');
        if (valueSpan) {
             // Format based on step (integer or float)
            const step = parseFloat(event.target.step);
            if (step === parseInt(step)) {
                valueSpan.textContent = parseInt(event.target.value);
            } else {
                 valueSpan.textContent = parseFloat(event.target.value).toFixed(step.toString().split('.')[1].length);
            }
        }
    }

    // Add event listeners to all range inputs for dynamic value display
    featureInputsDiv.querySelectorAll('input[type="range"]').forEach(input => {
        input.addEventListener('input', updateSliderValue);
    });


    predictBtn.addEventListener('click', function() {
        const selectedCrop = cropSelect.value;
        const featureValues = {};
        const featureInputs = featureInputsDiv.querySelectorAll('input[type="range"]');

        featureInputs.forEach(input => {
            featureValues[input.id] = parseFloat(input.value); // Parse as float
        });

        // Clear previous results
        predictionOutputDiv.textContent = 'Predicting...';
        shapPlotDiv.innerHTML = ''; // Clear previous plot
        recommendationsUl.innerHTML = '';

        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                crop: selectedCrop,
                features: featureValues
            }),
        })
        .then(response => {
            if (!response.ok) {
                // Handle HTTP errors
                return response.json().then(err => { throw new Error(err.error || `HTTP error! status: ${response.status}`); });
            }
            return response.json();
        })
        .then(data => {
            // Display prediction
            predictionOutputDiv.textContent = 'PREDICTED YIELD (' + selectedCrop + '): ' + data.prediction.toFixed(0) + ' kg/acre';

            // Render SHAP plot using Plotly.js from JSON data
            if (data.plotly_json) {
                const plotlyData = JSON.parse(data.plotly_json);
                Plotly.react('shap-plot', plotlyData.data, plotlyData.layout); // Use Plotly.react for potential updates
            } else {
                shapPlotDiv.textContent = 'SHAP plot data not available.';
            }


            // Display recommendations
            if (data.recommendations && data.recommendations.length > 0) {
                data.recommendations.forEach(rec => {
                    const li = document.createElement('li');
                    li.textContent = rec;
                    recommendationsUl.appendChild(li);
                });
            } else {
                 const li = document.createElement('li');
                 li.textContent = 'No specific recommendations at this time. Maintain current practices.';
                 recommendationsUl.appendChild(li);
            }

        })
        .catch((error) => {
            console.error('Error:', error);
            predictionOutputDiv.textContent = 'Error: ' + error.message; // Display error message
            shapPlotDiv.innerHTML = 'Error loading plot.';
            recommendationsUl.innerHTML = '';
        });
    });

     // Initial update of slider values on page load
     featureInputsDiv.querySelectorAll('input[type="range"]').forEach(input => {
        const valueSpan = document.getElementById(input.id + '-value');
         if (valueSpan) {
             // Format based on step (integer or float)
            const step = parseFloat(input.step);
            if (step === parseInt(step)) {
                valueSpan.textContent = parseInt(input.value);
            } else {
                 valueSpan.textContent = parseFloat(input.value).toFixed(step.toString().split('.')[1].length);
            }
        }
    });
});