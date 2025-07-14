import React from 'react';
import Layout from '@theme/Layout';
import clsx from 'clsx';
import styles from './index.module.css';

interface Talk {
  title: string;
  description: string;
  link: string;
  date?: string;
  platform: string;
  category: 'Technical' | 'Career' | 'Community' | 'Competition';
}

const talks: Talk[] = [
  {
    title: "Beyond CUDA",
    description: "Exploring the future of GPU computing beyond CUDA",
    link: "https://www.youtube.com/watch?v=RAK3Ce0RXgM",
    platform: "TensorWave",
    category: "Technical"
  },
  {
    title: "Infrastructure for Communities",
    description: "Building scalable infrastructure for AI communities",
    link: "https://www.youtube.com/watch?v=roztCMA86Z4",
    platform: "PrimeIntellect AI",
    category: "Community"
  },
  {
    title: "GPU MODE @ GTC",
    description: "Presenting GPU MODE at NVIDIA GTC",
    link: "https://www.youtube.com/watch?v=mdDVkBeFy9A",
    platform: "NVIDIA Developer",
    category: "Technical"
  },
  {
    title: "Why ML Systems Matter",
    description: "The importance of ML systems in modern AI",
    link: "https://www.youtube.com/watch?v=GCtOq-YIyyQ",
    platform: "YouTube",
    category: "Technical"
  },
  {
    title: "GPU Optimization Workshop",
    description: "Deep dive into GPU optimization techniques",
    link: "https://www.youtube.com/watch?v=v_q2JTIqE20",
    platform: "YouTube",
    category: "Technical"
  },
  {
    title: "Hugging Face Course",
    description: "Teaching PyTorch and ML fundamentals",
    link: "https://www.youtube.com/watch?v=6NTHvcXAl90",
    platform: "Hugging Face",
    category: "Technical"
  },
  {
    title: "Slaying OOM",
    description: "Strategies for handling out-of-memory issues in ML",
    link: "https://www.youtube.com/watch?v=UvRl4ansfCg",
    platform: "Hamel Husain",
    category: "Technical"
  },
  {
    title: "Performance of CUDA with PyTorch Flexibility",
    description: "Balancing performance and flexibility in deep learning",
    link: "https://www.nvidia.com/en-us/on-demand/session/gtc25-S71946/",
    platform: "NVIDIA GTC",
    category: "Technical"
  },
  {
    title: "Mojo: The Future of AI Programming",
    description: "Exploring Mojo's potential in AI development",
    link: "https://docs.google.com/presentation/d/1bGpvNxJKyS_ZMiVlpJTopXtQuoHQhqCq/edit",
    platform: "Google Slides",
    category: "Technical"
  },
  {
    title: "Dealing with Career Stagnation",
    description: "Navigating career growth in tech",
    link: "https://www.youtube.com/watch?v=5Ldn_nCPNNQ",
    platform: "USF Data Institute",
    category: "Career"
  },
  {
    title: "The Great ML Stagnation",
    description: "Discussion on ML Street Talk about industry challenges",
    link: "https://www.youtube.com/watch?v=BwhBtvCNwxo",
    platform: "ML Street Talk",
    category: "Technical"
  },
  {
    title: "Architecture of AI-Powered Drug Discovery",
    description: "Technical deep dive into AI drug discovery systems",
    link: "https://www.youtube.com/watch?v=fgB5-wyD-f0",
    platform: "Graphcore",
    category: "Technical"
  },
  {
    title: "The Power of Open Platforms",
    description: "Advocating for hardware-neutral tools in AI",
    link: "https://www.youtube.com/watch?v=iD4c25hy3Jc",
    platform: "AMD Developer Central",
    category: "Technical"
  },
  {
    title: "How to Train a Model with PyTorch",
    description: "Comprehensive guide to PyTorch model training",
    link: "https://www.youtube.com/watch?v=KmvPlW2cbIo",
    platform: "Hugging Face",
    category: "Technical"
  },
  {
    title: "PyTorch 2023 Keynote",
    description: "Keynote presentation at PyTorch Conference 2023",
    link: "https://youtu.be/98enl3C3fps",
    platform: "PyTorch",
    category: "Technical"
  },
  {
    title: "NeurIPS LLM Efficiency Challenge",
    description: "Competition focused on LLM efficiency",
    link: "https://neurips.cc/virtual/2023/competition/66594",
    platform: "NeurIPS",
    category: "Competition"
  },
  {
    title: "Hacker Cup AI Competition",
    description: "AI-focused competitive programming challenge",
    link: "https://neurips.cc/virtual/2024/competition/84789",
    platform: "NeurIPS",
    category: "Competition"
  }
];

const TalkCard: React.FC<{ talk: Talk }> = ({ talk }) => (
  <div className={clsx('card', styles.talkCard)}>
    <div className="card__header">
      <h3>{talk.title}</h3>
    </div>
    <div className="card__body">
      <p>{talk.description}</p>
      <div className={styles.talkMeta}>
        <span className={clsx('badge badge--primary', styles.platform)}>{talk.platform}</span>
        <span className={clsx('badge badge--secondary', styles.category)}>{talk.category}</span>
      </div>
    </div>
    <div className="card__footer">
      <a href={talk.link} target="_blank" rel="noopener noreferrer" className="button button--primary button--block">
        Watch Talk →
      </a>
    </div>
  </div>
);

export default function Talks(): JSX.Element {
  const categories = Array.from(new Set(talks.map(talk => talk.category)));
  
  return (
    <Layout
      title="Talks"
      description="Collection of talks and presentations by Mark Saroufim"
    >
      <header className={clsx('hero hero--primary', styles.heroBanner)}>
        <div className="container">
          <h1 className="hero__title">Talks & Presentations</h1>
          <p className="hero__subtitle">Collection of technical talks, career discussions, and community presentations</p>
        </div>
      </header>
      
      <main className="container">
        <div className={styles.talksContainer}>
          {categories.map(category => (
            <section key={category} className={styles.categorySection}>
              <h2 className={styles.categoryTitle}>{category}</h2>
              <div className={styles.talksGrid}>
                {talks
                  .filter(talk => talk.category === category)
                  .map((talk, idx) => (
                    <TalkCard key={idx} talk={talk} />
                  ))}
              </div>
            </section>
          ))}
        </div>
      </main>
    </Layout>
  );
} 