import { check } from 'k6';
import http from 'k6/http';
import { Trend } from 'k6/metrics';

// Define a custom trend metric for tracking response times
const myTrend = new Trend('custom_response_time');

export default function () {
  // Make a request to your endpoint
  const res = http.get("https://api.thecatapi.com/v1/images/search?limit=10");

  // Track response time in custom trend
  myTrend.add(res.timings.duration);

  // Check if the response status is 200 OK
  check(res, {
    'is status 200': (r) => r.status === 200,
  });

  // Log some information
  console.log('Response time: ' + res.timings.duration);
}
