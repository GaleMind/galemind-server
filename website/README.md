# The GaleMind documentation site

[Docusaurus](https://docusaurus.io/), serving the contents of `docs/`.

```bash
npm ci
npm start          # a dev server with live reload on http://localhost:3000
npm run build      # the static site, into build/
npm run serve      # serve what was built
npm run typecheck  # the TypeScript in src/
```

Node 20 or newer; `package.json` says so and the build refuses older versions.

## How it is put together

- `docs/` is the content. The sidebar is written out by hand in `sidebars.ts`
  rather than generated from the filesystem, because the order to read the
  pages in is not the order a directory listing happens to be in.
- `docusaurus.config.ts` holds the site's identity and the deployment URL. It
  is configured for GitHub Pages at `/galemind-server/`; serving it elsewhere
  means changing `url` and setting `baseUrl` to `/`.
- `onBrokenLinks` is `throw`. A link to a page that does not exist fails the
  build rather than shipping a dead link, which is the whole reason to build
  docs rather than write them.
- There is no blog. An empty blog is worse than no blog; `blog: false` in the
  preset turns it off, and removing that line brings it back.

## The architecture diagrams

The 4+1 views on the architecture page are Mermaid sources in `diagrams/`,
rendered to SVG in `static/img/views/` and committed. Rendering is a
deliberate step rather than a client-side library: the page then carries no
diagram runtime, the SVGs are searchable and diffable, and a diagram that
fails to render fails here instead of in a reader's browser.

```bash
npx @mermaid-js/mermaid-cli -i diagrams/logical-view.mmd \
  -o static/img/views/logical-view.svg -b transparent
```

All six at once:

```bash
for d in logical-view process-view development-view physical-view \
         scenario-infer scenario-memory; do
  npx @mermaid-js/mermaid-cli -i "diagrams/$d.mmd" \
    -o "static/img/views/$d.svg" -b transparent
done
```

mermaid-cli drives a headless Chromium. Where it cannot launch one -- a
container without the sandbox -- pass it a puppeteer config:

```bash
echo '{"args":["--no-sandbox","--disable-dev-shm-usage"]}' > /tmp/puppeteer.json
npx @mermaid-js/mermaid-cli -p /tmp/puppeteer.json -i ... -o ...
```

The diagrams are drawn for a light background and are wider than a page, so
the `.diagram` class in `src/css/custom.css` gives them a light card and lets
them scroll rather than shrinking the text to nothing. Edit the `.mmd`,
re-render, and commit both.

## Deploying

```bash
GIT_USER=<your-github-username> npm run deploy
```

That builds the site and pushes it to the `gh-pages` branch. Anything else
that can serve a directory of static files will serve `build/` as it is.

## Keeping it honest

Every number on these pages came from a measured run, and the page says which
machine and which command. When a measurement changes, the page it is on
changes with it — a documented figure nobody re-measures is how documentation
starts lying.
