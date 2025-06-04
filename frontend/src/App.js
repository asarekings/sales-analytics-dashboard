import React, { useState } from "react";
const demoData = [
  { month: "Jan", sales: 120 },
  { month: "Feb", sales: 180 },
  { month: "Mar", sales: 100 },
  { month: "Apr", sales: 80 },
  { month: "May", sales: 90 },
  { month: "Jun", sales: 150 },
];
function App() {
  const [data] = useState(demoData);
  return (
    <div style={{ maxWidth: 600, margin: "2em auto", background: "#fff", borderRadius: 8, boxShadow: "0 2px 8px rgba(0,0,0,0.07)", padding: "2em" }}>
      <h1>Sales Analytics Dashboard</h1>
      <table style={{ width: "100%", marginBottom: "2em" }}>
        <thead>
          <tr>
            <th>Month</th>
            <th>Sales</th>
          </tr>
        </thead>
        <tbody>
          {data.map(({ month, sales }) => (
            <tr key={month}>
              <td>{month}</td>
              <td>{sales}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <BarChart data={data} />
    </div>
  );
}
function BarChart({ data }) {
  const max = Math.max(...data.map(d => d.sales));
  const barWidth = 60, chartHeight = 200, chartWidth = data.length * barWidth;
  return (
    <svg width={chartWidth} height={chartHeight} style={{ background: "#f8f9fa", border: "1px solid #eee" }}>
      {data.map((d, i) => (
        <rect
          key={d.month}
          x={i * barWidth + 10}
          y={chartHeight - (d.sales / max) * (chartHeight - 30)}
          width={barWidth - 20}
          height={(d.sales / max) * (chartHeight - 30)}
          fill="#007bff"
        />
      ))}
      {data.map((d, i) => (
        <text
          key={d.month}
          x={i * barWidth + barWidth / 2}
          y={chartHeight - 5}
          textAnchor="middle"
          fill="#333"
          fontSize="14"
        >
          {d.month}
        </text>
      ))}
    </svg>
  );
}
export default App;
