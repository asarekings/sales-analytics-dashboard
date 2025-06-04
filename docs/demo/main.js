function showChart() {
  const canvas = document.getElementById('chart');
  canvas.style.display = 'block';
  const ctx = canvas.getContext('2d');
  // Simple bar chart
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const sales = [120, 180, 100, 80, 90, 150];
  const labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'];
  ctx.fillStyle = '#007bff';
  for (let i = 0; i < sales.length; i++) {
    ctx.fillRect(40 + i*50, 180 - sales[i], 30, sales[i]);
    ctx.fillStyle = '#333';
    ctx.fillText(labels[i], 40 + i*50, 195);
    ctx.fillStyle = '#007bff';
  }
}