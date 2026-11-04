import argparse
import json


def main():
    parser = argparse.ArgumentParser(description='Urban Traffic Intelligence')
    commands = parser.add_subparsers(dest='command', required=True)
    fetch = commands.add_parser('fetch', help='Copy the supplied CSV and download a pinned Traffic Monitor Lizard snapshot')
    fetch.add_argument('--volume', help='Path to the original Bangalore CSV')
    fetch.add_argument('--revision', help='Exact upstream commit SHA')
    train = commands.add_parser('train', help='Audit, engineer features, train, evaluate and save artifacts')
    train.add_argument('--dataset', choices=['volume', 'journey', 'estimator', 'all'], default='all')
    train.add_argument('--quick', action='store_true', help='Smaller tree ensemble for smoke testing')
    audit = commands.add_parser('audit', help='Inspect both datasets and their overlap')
    share = commands.add_parser('share', help='Run the app with a temporary public HTTPS link')
    share.add_argument('--port', type=int, default=8502, help='Local shared-app port (default: 8502)')
    share.add_argument('--cloudflared', help='Use an existing cloudflared executable instead of installing it')
    share.add_argument('--no-browser', action='store_true', help='Print the public link without opening a browser')
    args = parser.parse_args()
    if args.command == 'share':
        from .sharing import share_app
        raise SystemExit(share_app(port=args.port, cloudflared=args.cloudflared, open_browser=not args.no_browser))
    elif args.command == 'fetch':
        from .download import fetch_data
        kwargs = {'volume_path': args.volume}
        if args.revision:
            kwargs['revision'] = args.revision
        result = fetch_data(**kwargs)
    elif args.command == 'audit':
        from .data import audit_all
        result = audit_all()
    else:
        from .training import train_all
        result = train_all(args.dataset, quick=args.quick)
    print(json.dumps(result, indent=2, default=str, allow_nan=False))


if __name__ == '__main__':
    main()
