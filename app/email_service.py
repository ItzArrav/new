"""Email sending service supporting Gmail SMTP and simulated fallback mode."""
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from sqlalchemy.orm import Session

from .config import settings
from .models import SentEmail

log = logging.getLogger(__name__)

DEFAULT_TEMPLATES = {
    "interview": {
        "title": "Interview Invitation",
        "subject": "Invitation to Interview: {{role}} at {{company}}",
        "body": """Dear {{name}},

Thank you for your application for the {{role}} position at {{company}}.

We thoroughly reviewed your CV and qualifications, and our team was very impressed by your background and experience. We would like to invite you for an interview to discuss how your skills align with our upcoming projects.

Could you please share your availability over the next few business days for an initial 30-minute discussion?

We look forward to speaking with you!

Best regards,
{{recruiter}}
{{company}}
""",
    },
    "shortlist": {
        "title": "Application Shortlisted",
        "subject": "Application Update: {{role}} at {{company}}",
        "body": """Dear {{name}},

Thank you for taking the time to apply for the {{role}} role at {{company}}.

We are pleased to inform you that your resume has been shortlisted! Our hiring team is currently finalizing our interview schedule, and we will be in touch with you shortly regarding the next steps.

Thank you again for your patience and interest in {{company}}.

Warm regards,
{{recruiter}}
{{company}}
""",
    },
    "reject": {
        "title": "Respectful Rejection",
        "subject": "Update regarding your application for {{role}} at {{company}}",
        "body": """Dear {{name}},

Thank you for your interest in {{company}} and for taking the time to share your resume for the {{role}} opportunity.

We reviewed your application carefully. While your background and skills are commendable, we have decided to move forward with other candidates whose experience more closely matches our immediate technical requirements for this role.

We will keep your resume on file for future openings that match your profile. We truly appreciate your time and wish you the best in your career pursuits.

Sincerely,
{{recruiter}}
{{company}}
""",
    },
    "custom": {
        "title": "Custom Message",
        "subject": "Regarding your application for {{role}} at {{company}}",
        "body": """Dear {{name}},

Thank you for your interest in {{company}}.

[Write your custom message here]

Best regards,
{{recruiter}}
{{company}}
""",
    },
}


def render_template(template_str: str, replacements: dict[str, str]) -> str:
    """Replaces {{key}} placeholders in template strings."""
    result = template_str
    for key, val in replacements.items():
        result = result.replace(f"{{{{{key}}}}}", str(val or ""))
    return result


def send_email_message(
    to_email: str,
    subject: str,
    body: str,
    db: Session,
    user_id: int,
    candidate_name: str | None = None,
    recommendation_id: int | None = None,
    template_type: str = "custom",
) -> dict:
    """Sends an email via SMTP or records a simulation if SMTP is not configured."""
    clean_to = (to_email or "").strip()
    if not clean_to or "@" not in clean_to:
        raise ValueError("Invalid recipient email address")

    # Check if SMTP is configured (e.g. Gmail)
    has_smtp = bool(settings.smtp_host and settings.smtp_user and settings.smtp_password)
    sender = settings.smtp_from or settings.smtp_user or "recruiting@localhost"

    status = "sent"
    error_msg = None

    if has_smtp:
        try:
            msg = MIMEMultipart()
            msg["From"] = sender
            msg["To"] = clean_to
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain", "utf-8"))

            if settings.smtp_port == 465 or not settings.smtp_use_tls:
                server = smtplib.SMTP_SSL(settings.smtp_host, settings.smtp_port, timeout=12)
            else:
                server = smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=12)
                server.starttls()

            server.login(settings.smtp_user, settings.smtp_password)
            server.send_message(msg)
            server.quit()
            log.info("Email delivered to %s via %s", clean_to, settings.smtp_host)
        except Exception as e:
            log.exception("Failed to send email to %s", clean_to)
            status = "failed"
            error_msg = str(e)
    else:
        # Simulated mode: perfectly safe for demos, dev, and local tests
        status = "simulated"
        log.info(
            "[SIMULATED EMAIL] To: %s | Subject: %s | (Configure SMTP_HOST & SMTP_USER in .env for live Gmail)",
            clean_to,
            subject,
        )

    # Record in database
    record = SentEmail(
        user_id=user_id,
        recommendation_id=recommendation_id,
        recipient_email=clean_to,
        candidate_name=candidate_name,
        subject=subject,
        body=body,
        template_type=template_type,
        status=status,
        error=error_msg,
    )
    db.add(record)
    db.commit()

    return {
        "id": record.id,
        "recipient_email": clean_to,
        "candidate_name": candidate_name,
        "status": status,
        "is_simulated": status == "simulated",
        "error": error_msg,
        "subject": subject,
        "ts": record.ts.isoformat(),
    }
