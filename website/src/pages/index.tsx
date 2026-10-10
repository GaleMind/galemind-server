import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import CodeBlock from '@theme/CodeBlock';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import Heading from '@theme/Heading';

import styles from './index.module.css';

const quickstart = `make rocksdb-lib
cargo build --release
cargo run --release -p galemind -- serve

gmctl models
gmctl infer building_energy --rows 4`;

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero', styles.heroBanner)}>
      <div className="container">
        <div className="row">
          <div className={clsx('col col--6', styles.heroText)}>
            <Heading as="h1" className={styles.heroTitle}>
              {siteConfig.title}
            </Heading>
            <p className={styles.heroSubtitle}>{siteConfig.tagline}</p>
            <div className={styles.buttons}>
              <Link
                className="button button--primary button--lg"
                to="/docs/intro">
                Read the docs
              </Link>
              <Link
                className="button button--secondary button--lg"
                to="/docs/api/overview">
                The API
              </Link>
            </div>
          </div>
          <div className="col col--6">
            <CodeBlock language="bash" className={styles.heroCode}>
              {quickstart}
            </CodeBlock>
          </div>
        </div>
      </div>
    </header>
  );
}

/// The numbers are measured, and every one of them is explained in the docs.
function HomepageNumbers(): ReactNode {
  const numbers = [
    {
      figure: '10 012',
      label: 'sockets at once',
      detail: 'real connections from ten processes, nothing dropped',
      to: '/docs/operations/load-testing',
    },
    {
      figure: '48 684',
      label: 'requests/s',
      detail: 'at concurrency 10 000 over gRPC, on twelve cores',
      to: '/docs/operations/load-testing',
    },
    {
      figure: '17 MB',
      label: 'resident at rest',
      detail: 'and a plateau, not a climb, under load',
      to: '/docs/operations/memory',
    },
    {
      figure: '10',
      label: 'models served',
      detail: 'from a feature scaler to a 256×256 generator',
      to: '/docs/model-repository',
    },
  ];

  return (
    <section className={styles.numbers}>
      <div className="container">
        <div className="row">
          {numbers.map((number) => (
            <div className={clsx('col col--3')} key={number.figure}>
              <Link className={styles.number} to={number.to}>
                <span className={styles.numberFigure}>{number.figure}</span>
                <span className={styles.numberLabel}>{number.label}</span>
                <span className={styles.numberDetail}>{number.detail}</span>
              </Link>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title="A CPU inference server"
      description={siteConfig.tagline as string}>
      <HomepageHeader />
      <main>
        <HomepageNumbers />
        <HomepageFeatures />
      </main>
    </Layout>
  );
}
