const CACHE_NAME = 'laoliu-v7';

self.addEventListener('install', e => {
  self.skipWaiting();
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);

  // Static page assets: always network-first (never serve stale HTML/JS)
  if (url.origin === self.location.origin) {
    e.respondWith(
      fetch(e.request, { cache: 'no-store' })
        .then(resp => {
          if (resp.ok) {
            const copy = resp.clone();
            caches.open(CACHE_NAME).then(cache => cache.put(e.request, copy));
            return resp;
          }
          return caches.match(e.request);
        })
        .catch(() => caches.match(e.request))
    );
    return;
  }

  // External API data: stale-while-revalidate
  e.respondWith(
    caches.match(e.request).then(cached => {
      const networkPromise = fetch(e.request, { cache: 'no-store' })
        .then(resp => {
          if (resp.ok) {
            const copy = resp.clone();
            caches.open(CACHE_NAME).then(cache => cache.put(e.request, copy));
            return resp;
          }
          return cached;
        })
        .catch(() => cached);
      return cached || networkPromise;
    })
  );
});
