"""Isolated experimental service entry; baseline resident service unchanged."""
import asyncio
import json
import os
from resident_openpi_server import parse_args,create_policy,serve
from fastrsi.trial_fork_policy import TrialForkPolicy


def main():
    args=parse_args()
    native=create_policy(args)
    policy=TrialForkPolicy(native,source_session=os.environ['FASTRSI_TRIAL_SOURCE_SESSION'],
                           prefix_inferences=int(os.environ['FASTRSI_TRIAL_PREFIX_INFERENCES']),
                           max_sessions=args.max_sessions,max_inferences=args.max_inferences)
    args.receipt_dir.mkdir(parents=True,exist_ok=True)
    with (args.receipt_dir/'STARTUP.json').open('x') as f:
        json.dump(dict(metadata=policy.metadata,rng_hashes=policy.rng.hashes,
                       policy_state=policy.guard.reset(),source_session=policy.source_session,
                       prefix_inferences=policy.prefix_inferences),f,indent=2)
    asyncio.run(serve(policy,host=args.host,port=args.port,receipt_dir=args.receipt_dir))


if __name__=='__main__':main()
