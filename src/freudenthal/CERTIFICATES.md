# Fixed exact witnesses

These files are mathematical inputs for the computer-assisted construction
in Sections 3–4 of the accompanying manuscript. The compressed rule table
contains the higher-degree coefficient types, source equations and owners.
The matrix archive contains the degree-six normal form, local repair parts,
small-grid bases and prescribed minor records, plus finite coverage inputs.
It has 35 individually hashed members in `MANIFEST.json`.

| File | SHA-256 |
|---|---|
| `all_degree_rules.json.gz` | `43d45cfd9394befe130b7e5e27f22851369e0adaf0669f8526e3d8156bc7b5ee` |
| `verification_data.zip` | `32c3c6847241288a4607d699e68ac7fbbdae3378e4057531c44a94274f85f7c7` |

The library verifies these compressed files before use. The root
`verify_all.py` checks all extracted members, executes the supplied checking
programs, compares recomputed higher-degree owners and independently
reassembles the degree-six matrices with analytical integer gradients.

Source conditions use arbitrary-precision Python integers; dimension counts
use exact rational symbolic algebra. Sparse integer products are bounded
before evaluation. The C++17 affine checker is compiled with warnings as
errors and undefined-behavior sanitization. Section 4.2 gives conservative
signed-integer bounds. The full parameter-coverage arguments remain necessary:
successful finite mesh samples alone do not establish the uniform theorem.

The `_certificate_check/` sources preserve a distinct checker implementation,
including mathematical discovery helpers; discovery runs are not acceptance
predicates. Modular determinants are nonvanishing integer witnesses for real
rank lower bounds and are paired with exact membership. No equality between
arbitrary real and finite-field ranks is assumed.
