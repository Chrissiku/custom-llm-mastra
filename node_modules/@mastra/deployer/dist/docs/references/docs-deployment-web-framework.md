# Deploy with a web framework

When Mastra is integrated with a web framework, it deploys alongside your application using the framework's standard deployment process. Follow the instructions below to ensure your Mastra integration deploys correctly.

> **Warning:** If you're deploying to a cloud provider, remove any usage of [LibSQLStore](https://mastra.ai/reference/storage/libsql) from your Mastra configuration. LibSQLStore requires filesystem access and isn't compatible with serverless platforms.

Integration guides:

- [With Next.js](https://mastra.ai/guides/getting-started/next-js)
- [With Astro](https://mastra.ai/guides/getting-started/astro)

## With Next.js on Vercel

If you've integrated Mastra with Next.js [by following our guide](https://mastra.ai/guides/getting-started/next-js) and plan to deploy to Vercel, add `serverExternalPackages: ["@mastra/*"]` to your `next.config.ts`:

```typescript
import type { NextConfig } from 'next'

const nextConfig: NextConfig = {
  serverExternalPackages: ['@mastra/*'],
}

export default nextConfig
```

## With Astro on Vercel

If you've integrated Mastra with Astro [by following our guide](https://mastra.ai/guides/getting-started/astro) and plan to deploy to Vercel, add the Vercel adapter and server output to your `astro.config.mjs`:

```javascript
import { defineConfig } from 'astro/config'
import vercel from '@astrojs/vercel'

export default defineConfig({
  adapter: vercel(),
  output: 'server',
})
```

## With Astro on Netlify

If you've integrated Mastra with Astro [by following our guide](https://mastra.ai/guides/getting-started/astro) and plan to deploy to Netlify, add the Netlify adapter and server output to your `astro.config.mjs`:

```javascript
import { defineConfig } from 'astro/config'
import netlify from '@astrojs/netlify'

export default defineConfig({
  adapter: netlify(),
  output: 'server',
})
```