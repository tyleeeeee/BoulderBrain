function fetchData() {
    fetch('http://localhost:5000/api/data')
        .then(response => response.json())
        .then(data => {
            document.getElementById('data').innerText = data.message;
        })
        .catch(error => console.error('Error:', error));
}
