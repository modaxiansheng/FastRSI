"""Explicit infrastructure restart support; never retry native failures."""
import asyncio, json

def completed_case(folder, task, model):
 if not folder.exists():return None
 if (folder/'ERROR.json').exists() or not (folder/'RESULT.json').exists():
  raise ValueError('incomplete case must be explicitly archived before retry: '+str(folder))
 row=json.loads((folder/'RESULT.json').read_text())
 if row.get('task')!=task or row.get('model')!=model or row.get('execution_status')!='completed':
  raise ValueError('existing result identity/status mismatch: '+str(folder))
 return row

async def verify_async(verify, plan, episode, folder):
 return await asyncio.get_running_loop().run_in_executor(None,verify,plan,episode,folder)
