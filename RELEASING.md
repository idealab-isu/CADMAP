# Release procedure

This release uses main as the accepted source baseline, release/v0.2 as the
0.2 maintenance line, and an immutable v0.2.0 tag for the published snapshot.
The branch and tag refer to the same commit at publication. Separate fix and
feature branches are not included merely because they exist locally.

1. Review the intended changes, dependency compatibility and known limitations.
2. Update the version source and CHANGELOG.md. CADMap uses
   src/cadmap/_version.py; EigenFit derives builds from Git tags and retains
   its baseline in EigenFit/version.txt. Do not edit generated EigenFit/_version.py.
3. Run the available regression tests, build wheel and sdist, and check metadata
   with python -m twine check. Verify that MATLAB helpers, processtrak steps and
   third-party licenses are in the relevant wheels.
4. In an isolated Python environment with the approved scientific dependencies,
   install the actual wheels rather than editable source. Check runtime versions,
   CLI entry points, processtrak discovery and representative synthetic geometry.
   Record separately any licensed-solver and private-data acceptance tests.
5. Commit accepted changes to main, fast-forward release/v0.2, and create the
   annotated version tag. Build final artifacts from the clean tagged commit;
   EigenFit must have access to the tag when building from Git.
6. Verify the final artifacts and record their SHA-256 hashes. Push only the
   intended main, release branch and tag. Create a GitHub Release with the wheel,
   sdist, checksums and validation report, explicitly stating test limitations.
7. Confirm that GitHub refs and release assets match the local validated snapshot.

This documents the release procedure; it does not assert that automated CI/CD
or branch protection has been configured. Publishing a release does not update
HPC installations or another source checkout automatically. Later release fixes
must be reconciled with the development line before the next version.
