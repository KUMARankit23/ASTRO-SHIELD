document.getElementById('prediction-form').addEventListener('submit', function (e) {
    e.preventDefault();

    const formData = new FormData(this);
    const jsonData = {};

    for (let [key, value] of formData.entries()) {
        jsonData[key] = parseFloat(value);
    }

    fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(jsonData)
    })
        .then(response => response.json())
        .then(data => {
            if (data.prediction !== undefined) {
                document.getElementById('result').textContent = `Prediction: ${data.prediction}`;
            } else {
                document.getElementById('result').textContent = `Error: ${data.error}`;
            }
        })
        .catch(err => {
            document.getElementById('result').textContent = `Error: ${err.message}`;
        });
});
