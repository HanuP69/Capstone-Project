import { PageFrame, PageHeader, Panel } from '../components/ui';

const team = [
  ['Sujal Kumar Yadav', '2024UGCS033', 'Verification agent architecture and orchestration'],
  ['Tanishk S Pandey', '2024UGCS037', 'Answer generation and retrieval'],
  ['Mohit Anand', '2024UGCS072', 'Groundedness scoring and evaluation'],
  ['Ankit Sharma', '2024UGCS063', 'Fact check, policy agents and backend API'],
  ['Dhirendra Pratap Dindoure', '2024UGCS084', 'Web dashboard and results visualization'],
  ['Himanshu Gupta', '2024UGCS113', 'Dataset curation and human evaluation'],
];

export function AboutPage() {
  return <PageFrame>
    <PageHeader title="About the project" description="A computer science capstone project on checking academic assistant answers against records and policy." />
    <section className="project-intro"><p className="project-title">Multi-Agent Groundedness Verification Pipeline for Institutional Academic Assistants</p><p>We studied a campus question-answering assistant for NIT Jamshedpur. The system breaks a draft answer into claims, checks each claim, and keeps a record of its evidence and decision.</p></section>
    <div className="about-info-grid">
      <Panel title="Institution"><div className="institution-block"><strong>National Institute of Technology Jamshedpur</strong><span>Department of Computer Science &amp; Engineering</span><span>Supervisor: Dr. Dilip Kumar</span></div></Panel>
      <Panel title="Data and scope"><div className="disclosure-box"><p>The institutional results use a controlled benchmark built from published academic ordinances, fee schedules, examination timetables and hostel rules. It is not a live feed from the institute.</p><p>Ingesting and checking live circulars is future work.</p></div></Panel>
    </div>
    <Panel title="Project team" subtitle="Contributors and areas of work"><div className="team-table" id="team"><div className="team-header"><span>Name</span><span>Roll number</span><span>Area of work</span></div>{team.map(([name, roll, responsibility]) => <div className="team-row" key={roll}><span><strong>{name}</strong></span><span className="roll-number">{roll}</span><span>{responsibility}</span></div>)}</div></Panel>
  </PageFrame>;
}
