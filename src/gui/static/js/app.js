/**
 * CENTAUR Dashboard - Custom JavaScript
 *
 * Minimal JS for functionality not covered by HTMX/Alpine.
 */

// Configure HTMX
document.addEventListener('htmx:configRequest', (event) => {
    // Add any custom headers here if needed
});

// Handle HTMX errors
document.addEventListener('htmx:responseError', (event) => {
    console.error('HTMX Error:', event.detail);
    // Could show a toast notification here
});

// Log HTMX events in development
if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
    document.addEventListener('htmx:afterRequest', (event) => {
        console.log('HTMX request completed:', event.detail.pathInfo.requestPath);
    });
}
