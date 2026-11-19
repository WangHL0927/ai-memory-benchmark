import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

// The Python runner passes only a guarded connection string in a clean environment.
const url = new URL(process.env.DATABASE_URL);
const name = process.env.X2_EVAL_DB;
const port = process.env.X2_EVAL_DB_PORT;
if (!/^[a-z][a-z0-9_]*_eval$/.test(name ?? '') || !/^\d+$/.test(port ?? '')) {
  throw new Error('Missing dedicated evaluation database parameters');
}
if (!['postgres:', 'postgresql:'].includes(url.protocol) ||
    url.hostname !== '127.0.0.1' || url.port !== port ||
    url.pathname !== '/' + name || url.search || url.hash ||
    !new RegExp('^' + name + '_[0-9a-f]{12}$').test(url.username) || !/^[0-9a-f]{64}$/.test(url.password)) {
  throw new Error('Expected the locally provisioned X2 evaluation URL');
}

const [root, action] = process.argv.slice(2);
const input = JSON.parse(readFileSync(0, 'utf8'));
const source = name => import(pathToFileURL(join(root, 'server', 'src', name + '.ts')).href);
const { createPool, database } = await source('db');
const pool = createPool(url.href);
const db = database(pool);
try {
  const identity = await db.one('select current_database() as db, current_user as role');
  if (identity.db !== name || identity.role !== url.username) throw new Error('Database identity mismatch');
  let result;
  if (action === 'prepare') {
    const { migrate } = await source('migrate');
    const migrations = await migrate(db);
    // Deactivate interrupted evaluation samples before another review proxy starts.
    const stale = await db.query(`
      update agent a set disabled_at = now() from tenant t
      where a.tenant_id = t.id and t.name ~ '^x2-eval-[0-9a-f]{32}$' and a.disabled_at is null
      and exists (select 1 from draft d join api_key k on k.id = d.submitter
        where k.tenant_id = t.id and k.principal_id = a.id and k.principal_type = 'agent'
        and d.state = 'pending')
      returning a.id`);
    result = { database: identity.db, migrations, interrupted: stale.length };
  } else if (action === 'tenant') {
    const { createAgent, setAgentGrant } = await source('agents');
    const { createKey } = await source('auth');
    const { access } = await source('access');
    result = await db.tx(async client => {
      const { rows: [tenant] } = await client.query(
        'insert into tenant(name) values ($1) returning id', [input.tenant]);
      const { rows: [user] } = await client.query(
        'insert into app_user(tenant_id, username) values ($1, $2) returning id', [tenant.id, 'eval-user']);
      await client.query("insert into space(tenant_id, kind, name) values ($1, 'user', 'eval-user')", [tenant.id]);
      const actor = { ...user, tenant_id: tenant.id, name: 'eval-user', principal_type: 'user' };
      const agent = await createAgent(client, actor, 'eval-agent');
      const { rows: [global] } = await client.query(
        "select id from space where tenant_id = $1 and kind = 'global'", [tenant.id]);
      await setAgentGrant(client, actor, agent.id, global.id, 'write');
      const teams = [];
      for (const project of [...new Set(input.projects)]) {
        await client.query('insert into project(tenant_id, name) values ($1, $2)', [tenant.id, project]);
        const { rows: [team] } = await client.query(
          "insert into space(tenant_id, kind, name) values ($1, 'team', $2) returning id", [tenant.id, project]);
        await client.query(
          "insert into space_grant(space_id, principal_type, principal_id, access) values ($1, 'user', $2, 'write')",
          [team.id, user.id]);
        await setAgentGrant(client, actor, agent.id, team.id, 'write');
        teams.push(team.id);
      }
      const rights = await access(client, { ...agent, tenant_id: tenant.id, principal_type: 'agent' });
      if (!teams.every(id => rights.write_space_ids.includes(id))) throw new Error('Missing evaluation team write grant');
      const key = await createKey(client, 'eval-key', { ...agent, tenant_id: tenant.id, principal_type: 'agent' });
      return { tenant_id: tenant.id, agent_id: agent.id, key: key.key };
    });
  } else if (action === 'pending') {
    result = await db.query(`
      select d.id, d.last_review_error, d.review_attempts
      from draft d join space s on s.id = d.space_id
      where s.tenant_id = $1 and d.state = 'pending' order by d.created_at, d.id`, [input.tenant_id]);
  } else if (action === 'export') {
    result = await db.tx(async client => {
      await client.query('set transaction isolation level repeatable read read only');
      const drafts = await client.query(`
        select d.id, d.op, d.proposed, d.session_id, d.evidence, d.state, d.decision, d.last_review_error, d.review_attempts
        from draft d join space s on s.id = d.space_id
        where s.tenant_id = $1 order by d.created_at, d.id`, [input.tenant_id]);
      const pages = await client.query(`
        select p.id, p.path, p.kind, p.title, p.body, p.status, p.rev, r.draft_id, s.kind as space_kind
        from page p join space s on s.id = p.space_id
        join revision r on r.page_id = p.id and r.rev = p.rev
        where s.tenant_id = $1 and p.status <> 'deleted'
        order by s.kind, s.name, p.path, p.id`, [input.tenant_id]);
      return { drafts: drafts.rows, pages: pages.rows };
    });
  } else if (action === 'disable') {
    await db.query('update agent set disabled_at = now() where tenant_id = $1 and id = $2',
      [input.tenant_id, input.agent_id]);
    result = true;
  } else {
    throw new Error('Unknown evaluation database action');
  }
  process.stdout.write(JSON.stringify(result));
} finally {
  await pool.end();
}
