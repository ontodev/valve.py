#!/usr/bin/env python3.9

from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("NEW_VERSION")
    parser.add_argument("CARGO_FILE")
    args = parser.parse_args()

    in_package_block = False
    cargo_file = open(args.CARGO_FILE)
    for line in cargo_file.readlines():
        line = line.rstrip()
        if not in_package_block:
            if line == "[package]":
                in_package_block = True
        else:
            if line.startswith("["):
                in_package_block = False
            else:
                if line.startswith("version"):
                    line = 'version = "{}"'.format(args.NEW_VERSION)
        print(line)
