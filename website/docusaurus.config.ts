import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const organizationName = 'GaleMind';
const projectName = 'galemind-server';
const repository = `https://github.com/${organizationName}/${projectName}`;

const config: Config = {
  title: 'GaleMind Server',
  tagline: 'A CPU inference server that speaks Predict Protocol v2, and nothing else',
  favicon: 'img/favicon.ico',

  // Future flags, see https://docusaurus.io/docs/api/docusaurus-config#future
  future: {
    v4: true, // Improve compatibility with the upcoming Docusaurus v4
  },

  // GitHub Pages serves the project site under /<projectName>/. Serving it
  // from somewhere else means changing url and setting baseUrl to '/'.
  url: `https://${organizationName.toLowerCase()}.github.io`,
  baseUrl: `/${projectName}/`,
  organizationName,
  projectName,
  deploymentBranch: 'gh-pages',
  trailingSlash: false,

  // A broken link is a documentation bug, so it fails the build rather than
  // shipping a dead page.
  onBrokenLinks: 'throw',
  markdown: {
    hooks: {
      onBrokenMarkdownLinks: 'throw',
    },
  },

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          editUrl: `${repository}/tree/main/website/`,
          showLastUpdateTime: true,
        },
        // No blog: this site documents a server, and an empty blog is worse
        // than no blog. Re-enable it here if that changes.
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    colorMode: {
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'GaleMind Server',
      logo: {
        alt: 'GaleMind',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'docs',
          position: 'left',
          label: 'Documentation',
        },
        {to: '/docs/api/overview', label: 'API', position: 'left'},
        {to: '/docs/operations/metrics', label: 'Metrics', position: 'left'},
        {
          href: repository,
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Documentation',
          items: [
            {label: 'Introduction', to: '/docs/intro'},
            {label: 'Installation', to: '/docs/installation'},
            {label: 'Running the server', to: '/docs/running-the-server'},
          ],
        },
        {
          title: 'Reference',
          items: [
            {label: 'Predict Protocol v2', to: '/docs/api/overview'},
            {label: 'The catalogue', to: '/docs/api/catalogue'},
            {label: 'Metrics', to: '/docs/operations/metrics'},
          ],
        },
        {
          title: 'More',
          items: [
            {label: 'GitHub', href: repository},
            {
              label: 'Open Inference Protocol',
              href: 'https://github.com/kserve/open-inference-protocol',
            },
            {label: 'License', href: `${repository}/blob/main/LICENSE`},
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} GaleMind. Apache 2.0 licensed.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: [
        'rust',
        'bash',
        'json',
        'toml',
        'python',
        'protobuf',
        'docker',
      ],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
