"""Generate a HARDER synthetic labelled CV set (40 CVs) to complement the easy one.

    python -m eval.make_hard_dataset
    python -m eval.run_eval --set hard

What makes it harder than eval/make_dataset.py:
- Neutral job titles ("Software Engineer", "Analyst" ...) for every CV, so the title never gives the label away.
- 8 mixed profiles with a real secondary category (for example back-end work plus some AI work).
- 8 "near miss" Other profiles that use the vocabulary of a category without doing that job: DevOps (vs back-end),
  manual QA (vs web automation), data analyst (vs AI), UX designer (vs front-end), cabin crew (vs PSS/DCS words),
  IT project manager (vs mobile/QA words), technical writer (vs APIs), IT support.
It is still synthetic: written by us, labelled by construction. Real anonymised CVs remain the stronger test.
"""
import csv
import random
from pathlib import Path

from eval.make_dataset import COMPANIES, FIRST, GENERIC, LAST, POOLS

OUT = Path("eval/data/hard")
NEUTRAL_TITLES = ["Software Engineer", "Engineer", "Consultant", "Technical Specialist", "Senior Engineer", "Analyst"]

MIXED = [  # (primary, secondary)
    ("Back-End Developer", "AI Engineer"), ("Front-End Developer", "Back-End Developer"),
    ("Web Automation", "Mobile Automation"), ("Mobile Automation", "Web Automation"),
    ("AI Engineer", "Chatbot Application Tester"), ("Chatbot Application Tester", "AI Engineer"),
    ("PSS", "DCS"), ("DCS", "PSS"),
]

NEAR_MISS_OTHER = [  # (what it is close to, skills, bullets) - all labelled Other
    ("DevOps (near Back-End)", "Kubernetes, Docker, Terraform, AWS, Jenkins, Prometheus, Ansible, Bash", [
        "Built CI/CD pipelines in Jenkins and GitHub Actions for 40 microservices",
        "Managed Kubernetes clusters on AWS EKS provisioned with Terraform",
        "Set up Prometheus and Grafana monitoring with on-call alerting",
        "Automated server configuration with Ansible playbooks",
        "Reduced cloud spend by 25 percent by right-sizing EC2 instances"]),
    ("Manual QA (near Web Automation)", "test cases, Jira, TestRail, UAT, regression testing, SQL", [
        "Wrote and executed manual test cases for a retail banking web portal",
        "Logged and tracked defects in Jira and verified fixes",
        "Ran regression and user acceptance testing before every release",
        "Reviewed requirements and wrote test plans with business analysts",
        "Checked responsive layouts by hand across Chrome, Safari and Edge"]),
    ("Data analyst (near AI Engineer)", "SQL, Power BI, Excel, Python, pandas, Snowflake", [
        "Built Power BI dashboards for sales, churn and regional performance",
        "Wrote SQL queries on Snowflake for weekly KPI reporting",
        "Cleaned and analysed survey data in Python with pandas",
        "Analysed A/B tests for marketing campaigns and presented the results",
        "Presented monthly insights to the leadership team"]),
    ("UX designer (near Front-End)", "Figma, user research, wireframing, prototyping, design systems", [
        "Designed wireframes and high-fidelity prototypes in Figma for a booking app",
        "Ran usability tests with 30 customers and turned findings into design changes",
        "Maintained the component library of the company design system in Figma",
        "Handed off annotated designs to the React development team",
        "Defined accessibility guidelines for colour contrast and font sizes"]),
    ("Cabin crew (near PSS/DCS vocabulary)", "passenger service, safety procedures, first aid, onboard sales", [
        "Delivered safety demonstrations and in-flight service on A320 flights",
        "Assisted passengers with reduced mobility during boarding and disembarkation",
        "Reconciled onboard sales and duty-free inventory after each sector",
        "Handled passenger complaints and wrote incident reports",
        "Completed annual emergency procedures and first aid recertification"]),
    ("IT project manager (near Mobile/QA words)", "Scrum, Jira, stakeholder management, budgeting, risk management", [
        "Managed delivery of a mobile banking app with a 12-person team",
        "Ran Scrum ceremonies and tracked sprint velocity in Jira",
        "Coordinated user acceptance testing between QA and the business",
        "Managed vendor contracts and a budget of 1.2 million dollars",
        "Reported project risks and status to the steering committee"]),
    ("Technical writer (near Back-End words)", "API documentation, Markdown, Swagger, user guides, style guides", [
        "Wrote reference documentation for 60 REST API endpoints from Swagger specifications",
        "Produced user guides and release notes for a payments platform",
        "Interviewed developers to document authentication and error-handling behaviour",
        "Maintained the documentation style guide and review process",
        "Reduced support tickets by 15 percent with clearer onboarding guides"]),
    ("IT support (no category)", "Active Directory, Windows, Office 365, VPN, ticketing, hardware", [
        "Resolved 40 or more helpdesk tickets a day for laptops, email and VPN",
        "Managed user accounts and group policies in Active Directory",
        "Imaged and deployed Windows laptops for new joiners",
        "Wrote knowledge-base articles for common issues",
        "Supported office moves and meeting-room hardware"]),
]


def _cv(title: str, skills: str, first_job: list[str], second_job: list[str], rng: random.Random) -> str:
    name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    c1, c2 = rng.sample(COMPANIES, 2)
    return (
        f"{name}\n{title}\n{name.split()[0].lower()}@example.com\n\n"
        f"SUMMARY\n{title} with {rng.randint(3, 11)} years of experience.\n\n"
        f"EXPERIENCE\n{c1} - {title} (2021 - present)\n" + "\n".join(f"- {b}" for b in first_job) +
        f"\n\n{c2} - {title} (2016 - 2021)\n" + "\n".join(f"- {b}" for b in second_job) +
        f"\n\nSKILLS\n{skills}\n\nEDUCATION\nB.Tech / B.Sc, {rng.choice(['Pune University', 'Anna University', 'Delhi University'])}\n"
    )


def _skills(pool_skills: str, rng: random.Random, k: int) -> list[str]:
    items = [s.strip() for s in pool_skills.split(",")]
    return rng.sample(items, min(k, len(items)))


def slug(cat: str) -> str:
    return cat.lower().replace(" ", "_").replace("-", "")


def main():
    (OUT / "cvs").mkdir(parents=True, exist_ok=True)
    rows = []
    for cat, pool in POOLS.items():  # 3 single-category CVs each, neutral titles, partial skills lists
        for i in range(3):
            rng = random.Random(f"hard-{cat}-{i}")
            bullets = rng.sample(pool["bullets"], 5)
            skills = ", ".join(_skills(pool["skills"], rng, 4))
            text = _cv(rng.choice(NEUTRAL_TITLES), skills, bullets[:3], bullets[3:] + [rng.choice(GENERIC)], rng)
            fn = f"hard_{slug(cat)}_{i + 1:02d}.txt"
            (OUT / "cvs" / fn).write_text(text, encoding="utf-8")
            rows.append((fn, cat, "", "normal", ""))
    for i, (primary, secondary) in enumerate(MIXED):  # 4 bullets of the primary area, 2 of the secondary
        rng = random.Random(f"hard-mixed-{i}")
        main_b = rng.sample(POOLS[primary]["bullets"], 4)
        sec_b = rng.sample(POOLS[secondary]["bullets"], 2)
        skills = ", ".join(_skills(POOLS[primary]["skills"], rng, 4) + _skills(POOLS[secondary]["skills"], rng, 2))
        text = _cv(rng.choice(NEUTRAL_TITLES), skills, main_b[:3] + sec_b[:1], main_b[3:] + sec_b[1:], rng)
        fn = f"hard_mixed_{slug(primary)}__{slug(secondary)}.txt"
        (OUT / "cvs" / fn).write_text(text, encoding="utf-8")
        rows.append((fn, primary, secondary, "normal", ""))
    for i, (_, skills, bullets) in enumerate(NEAR_MISS_OTHER):
        rng = random.Random(f"hard-other-{i}")
        text = _cv(rng.choice(NEUTRAL_TITLES), skills, bullets[:3], bullets[3:], rng)
        fn = f"hard_other_{i + 1:02d}.txt"
        (OUT / "cvs" / fn).write_text(text, encoding="utf-8")
        rows.append((fn, "Other", "", "normal", ""))

    with open(OUT / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "label", "secondary", "kind", "injection_target"])
        w.writerows(rows)
    print(f"Wrote {len(rows)} CVs to {OUT / 'cvs'} and {OUT / 'labels.csv'}")


if __name__ == "__main__":
    main()
