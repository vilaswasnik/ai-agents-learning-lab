# Lesson 04 — Copilot Studio: the same agent, built by clicking

Copilot Studio is Microsoft's **no-code** agent builder. There is no
Python to write and nothing to pip-install — you build in a browser,
and the agent lives inside Microsoft 365 / Teams. That is why this
lesson is a guide instead of code: it is the business version of the
same triage agent you built with LangGraph in lesson 02.

## When would you choose this over code?

- The users are inside Teams/Outlook all day and the agent should be too.
- The company is a Microsoft shop (Azure OpenAI, M365 licences already paid).
- A business team must maintain the agent without developers.

## Build the triage agent in Copilot Studio (about 30 minutes)

You need: a Microsoft 365 account with a Copilot Studio licence
(a trial works) — sign in at https://copilotstudio.microsoft.com

1. **Create the agent**
   - "Create" → "New agent" → name it `Server Triage Assistant`.
   - Description: "Helps the operations team triage Linux server incidents."

2. **Give it knowledge**
   - In the agent, open **Knowledge** → add your runbooks: upload PDFs/Word
     docs or point it at a SharePoint folder. This is Copilot Studio's
     version of "the agent reads the facts first" (lesson 02's CHECK node).

3. **Build the flow as a Topic** (a Topic = one conversation path,
   the no-code cousin of a LangGraph graph)
   - **Topics** → "Add a topic" → "From blank" → name it `Disk full triage`.
   - Trigger phrases: `disk full`, `filesystem full`, `server disk issue`.
   - Add a **Question** node: "Which server, and what exactly do you see?"
     — save the answer to a variable (this is STATE, like TriageState).
   - Add a **Condition** node on the answer: contains "disk" → disk branch;
     contains "cpu" → cpu branch (same routing idea as lesson 02's
     conditional edge).

4. **Add the approval step** (the guardrail — do not skip this)
   - Insert an **Adaptive Card** node showing the proposed plan with
     **Approve / Reject** buttons.
   - Only the Approve branch continues. This is lesson 02's APPROVAL node,
     built with two clicks instead of code.

5. **Call an action**
   - **Actions** → add a Power Automate flow or a connector (e.g. send a
     Teams message to the ops channel, create a ServiceNow incident).
   - This is the EXECUTE node. Start with something harmless like the
     Teams notification — never wire a real fix to a button on day one.

6. **Test and publish**
   - Use the built-in test chat pane on the right; walk the disk-full path.
   - **Publish**, then **Channels → Microsoft Teams** so the team can use it
     where they already work.

## Map it back to the code lessons

| Copilot Studio | LangGraph (lesson 02) |
|---|---|
| Topic | the graph |
| Variable | state (TriageState) |
| Condition node | conditional edge |
| Adaptive Card approval | approval node |
| Action / connector | tool / execute node |
| Knowledge sources | check node's facts |

Same agent thinking — different workshop. Code when you need control
and portability (any cloud, any model); Copilot Studio when the agent
must live inside Microsoft 365 and be maintained by non-developers.
