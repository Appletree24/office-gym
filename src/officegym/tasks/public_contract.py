"""Public business requirements; never reference actions or hidden outcomes."""

def completion_requirements(public):
    requirements=dict(meeting='Schedule the requested title for all required participants inside allowed_window_utc, lasting duration_seconds, avoiding calendar conflicts for every participant. A calendar draft is not completion: send a formal mail.send invitation to every participant with the current meeting_id and meeting_version and approval_id when approval is required. Meeting status changes automatically when invitations are sent; do not patch status.')
    if public.get('requires_approval'):requirements['approval']='Before formal external mail, obtain a granted approval for the exact current meeting version from the required approver. A request is not a grant. Updating the meeting invalidates old approvals.'
    if public.get('collect_availability'):
        requirements['availability']='Obtain actual participant availability replies using mail.send kind=coordination with request.type=availability, then inspect the replies. calendar.query alone is not a participant reply.'
        if public.get('proposal_bound_availability'):requirements['availability']+=' Bind both the request object and mail envelope to the current meeting_id/meeting_version, and recollect after a version change.'
        if public.get('approval_after_availability'):requirements['availability']+=' Request approval only after the current version has valid replies from all required participants.'
    if public.get('required_document_refs') or public.get('track_document_ids'):requirements['materials']='Bind all required/current tracked document versions in meeting.document_refs and formal-mail attachment_refs. Revisions require a new meeting version and, when required, a new approval.'
    if public.get('monitor_until'):requirements['monitoring']='Continue through monitor_until, observe public events and honor requested time/participant/document changes. Send reschedule mail after required corrections. Then declare completion with task.finish.'
    if public.get('followup_required'):requirements['followup']='After the meeting ends, read actual minutes and create the required action-item todos with correct owners and deadlines. Honor later public owner changes through followup_monitor_until when present.'
    if public.get('preparation_todos'):requirements['preparation']='Complete the listed preparation todos before requesting meeting approval.'
    if public.get('handoff_thread_id'):requirements['handoff']='Use the public handoff thread to identify the current coordinator/approver.'
    return requirements
