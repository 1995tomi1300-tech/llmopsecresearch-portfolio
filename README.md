# LLM OpSec Research Portfolio

> **Steckmüller Tamás Simon** — Independent Security Researcher

A minimal, Gothic-inspired portfolio website showcasing security research, bug bounty findings, and expertise in cryptographic vulnerabilities.

## 🚀 Quick Start

### Local Development

```bash
cd portfolio
npm install
npm run dev
```

This starts a development server with:
- Auto-reloading CSS changes
- JavaScript minification on save
- Concurrent processing

### Build for Production

```bash
cd portfolio
npm run build
```

This generates minified CSS and JavaScript files ready for deployment.

## 📁 Project Structure

```
llmopsecresearch-portfolio/
├── portfolio/
│   ├── index.html          # Main HTML entry point
│   ├── styles/
│   │   └── main.css        # All styles (modular CSS)
│   ├── scripts/
│   │   └── main.js         # JavaScript (cursor effects, etc.)
│   ├── images/
│   │   └── rose-window.svg # Optimized SVG graphic
│   ├── robots.txt          # SEO: Crawler instructions
│   ├── sitemap.xml         # SEO: Site structure
│   └── pgp-key.txt         # Public PGP key
├── .github/
│   └── workflows/
│       └── deploy.yml      # Cloudflare Pages CI/CD
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

## ✨ Features

- **Gothic Aesthetic** — Rose window animation, candle glow cursor
- **Fully Responsive** — Works on mobile and desktop
- **Performance Optimized** — Font preloading, minified assets
- **Accessible** — Skip links, reduced motion support, semantic HTML
- **SEO Ready** — OpenGraph, Twitter Cards, structured data
- **Modular** — Separated HTML, CSS, and JavaScript

## 🔧 Configuration

### Update Your Information

Edit these files to customize your portfolio:

1. **`portfolio/index.html`** — Update name, bio, findings, links
2. **`portfolio/pgp-key.txt`** — Add your actual PGP public key
3. **`portfolio/scripts/main.js`** — Modify cursor behavior
4. **`portfolio/styles/main.css`** — Customize colors and styling

### Platform Links

Update the social/platform links in `index.html`:
- Bugcrowd profile URL
- HackerOne profile URL  
- GitHub profile URL
- Email address (obfuscated in JavaScript)
- PGP fingerprint

## 🌍 Deployment

### Cloudflare Pages (Recommended)

1. Push to GitHub
2. In Cloudflare Dashboard:
   - Create new Pages project
   - Connect to GitHub repository
   - Set build command: `cd portfolio && npm run build`
   - Set build output: `portfolio`

### GitHub Pages

```bash
cd portfolio
npm run build
git add .
git commit -m "Build for deployment"
git push
```

### Netlify / Vercel

Both support automatic deployment from GitHub. No additional configuration needed.

## 🎨 Customization

### Colors

Edit CSS variables in `styles/main.css`:

```css
:root {
  --parchment: #f0e8d8;      /* Light text */
  --gold: #c9a84c;           /* Accent color */
  --ink: #1a1612;            /* Background */
  /* ... */
}
```

### Typography

The site uses:
- **IM Fell English** — Serif headings (Google Fonts)
- **Inter** — Sans-serif body text (Google Fonts)

Change fonts in the `<head>` of `index.html`.

### Animations

Disable animations by removing or commenting out:
- `@keyframes` rules in `main.css`
- Animation properties on elements
- The cursor glow JavaScript in `main.js`

## 🔍 SEO

The site includes:
- Semantic HTML5
- OpenGraph meta tags
- Twitter Card meta tags
- JSON-LD structured data
- `robots.txt` and `sitemap.xml`
- Canonical URL

For better SEO:
1. Add your actual profile URLs
2. Update the structured data with your real information
3. Submit `sitemap.xml` to Google Search Console

## 🛡️ Security

- All external links use `rel="noopener noreferrer"`
- Email is obfuscated via JavaScript
- Consider adding a Content Security Policy (CSP) header

## 📊 Performance

### Current Metrics (Estimated)

| Metric | Value |
|--------|-------|
| Page Weight | ~12-14KB |
| First Contentful Paint | ~0.8s |
| Largest Contentful Paint | ~1.5s |
| Time to Interactive | ~1.8s |

### Optimization Techniques Used

- ✅ Font preconnect and preload
- ✅ Minified SVG
- ✅ Modular CSS/JS (better caching)
- ✅ No render-blocking resources
- ✅ Efficient CSS selectors

## 🤝 Contributing

This is a personal portfolio. Contributions are welcome for:
- Bug fixes
- Accessibility improvements
- Performance optimizations

## 📄 License

MIT License — Feel free to use this template for your own portfolio.

---

**Built with** ❤️ **by Steckmüller Tamás Simon**
