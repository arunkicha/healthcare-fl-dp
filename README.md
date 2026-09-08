# Risk-Aware Privacy Preservation for Real-Time Health Analytics in EHR Systems

CSE4068 -- Arunkumaran A (23MIA1171)

## Status
- [ ] Dataset acquired (MIMIC-III Clinical Database Demo 1.4)
- [ ] Tier mapping defined and audited
- [ ] Client partitioning audited
- [ ] Centralized baseline
- [ ] Federated (no privacy)
- [ ] Federated + tiered DP

## Order of operations
1. Drop demo CSVs into `data/raw/mimic-iii-clinical-database-demo-1.4/`
2. Run audit scripts (audit_icd_coverage.py -> audit_tier_mapping.py -> audit_class_balance.py) BEFORE writing any model code
3. Only once audits look sane: build dataset.py -> partition_clients.py -> audit_client_distribution.py
4. Then baseline.py -> federated.py -> dp_federated.py
