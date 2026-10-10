import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  to: string;
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'One protocol, two transports',
    to: '/docs/api/overview',
    description: (
      <>
        The eight URLs of Predict Protocol v2 over HTTP and gRPC, plus two
        catalogue endpoints for the questions the protocol forgot: what can
        this server serve, and which versions of this model are there.
      </>
    ),
  },
  {
    title: 'Batched, cached, bounded',
    to: '/docs/running-the-server',
    description: (
      <>
        Concurrent requests for a model are merged into one call inside a
        window bounded at both ends, hot models stay loaded in a cache with a
        capacity and a time to live, and a full queue is back-pressure rather
        than a crash.
      </>
    ),
  },
  {
    title: 'Measured, not asserted',
    to: '/docs/operations/memory',
    description: (
      <>
        Every number in these docs came from a run on a twelve-core machine,
        including the ones that contradicted the first guess — the memory was
        glibc&apos;s arenas, and the C10K ceiling was the kernel&apos;s accept
        queue.
      </>
    ),
  },
  {
    title: 'Followable requests',
    to: '/docs/operations/tracing',
    description: (
      <>
        Every request gets an id, from <code>X-Request-Id</code> or minted, and
        a span that everything logged while handling it nests under — on both
        transports, and silent at <code>info</code>.
      </>
    ),
  },
  {
    title: 'Seams, not frameworks',
    to: '/docs/architecture',
    description: (
      <>
        The backend, the store, the cache, the clock and the identity provider
        are traits in a crate that does no I/O. Adding Keycloak or another
        runtime means implementing one of them, not editing the server.
      </>
    ),
  },
  {
    title: 'Prometheus, Triton-shaped',
    to: '/docs/operations/metrics',
    description: (
      <>
        Names follow Triton&apos;s with <code>gm_</code> for <code>nv_</code>,
        so dashboards port across with a prefix change. What cannot mean
        anything on a CPU server is left out rather than reported as zero.
      </>
    ),
  },
];

function Feature({title, to, description}: FeatureItem) {
  return (
    <div className={clsx('col col--4', styles.featureColumn)}>
      <Link className={styles.feature} to={to}>
        <Heading as="h3" className={styles.featureTitle}>
          {title}
        </Heading>
        <p className={styles.featureDescription}>{description}</p>
      </Link>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props) => (
            <Feature key={props.title} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
