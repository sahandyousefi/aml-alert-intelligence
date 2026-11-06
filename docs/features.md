# Feature dictionary

Generated from `src/aml_alert_intelligence/features/registry.py` by `scripts/make_feature_docs.py`. Do not edit by hand.

19 features. Flags are 0 or 1. No feature uses the label, a calendar date, or an account identifier, and none is fitted on anything but training days.

## A. Transaction

| Feature | Type | Label | Definition |
| --- | --- | --- | --- |
| `amount_usd_log` | numeric | Amount (USD, log10) | log10(1 + amount paid in US dollars), converted with the simulator's fixed exchange rates fitted on training days |
| `payment_format` | categorical | Payment format | Payment method: ACH, Bitcoin, Cash, Cheque, Credit Card, Reinvestment or Wire |
| `payment_currency` | categorical | Payment currency | Currency the sender pays in |
| `receiving_currency` | categorical | Receiving currency | Currency the receiver is paid in |
| `is_cross_currency` | flag | Cross-currency transfer | 1 if the two sides use different currencies |
| `is_self_transfer` | flag | Transfer to the same account | 1 if sender and receiver are the same account |
| `is_same_bank` | flag | Same bank | 1 if both accounts are at the same bank |
| `hour_of_day` | numeric | Hour of day | Hour of the transfer, 0 to 23, as recorded |
| `is_weekend` | flag | Weekend | 1 on Saturday or Sunday; replaces day of week, which would identify single dates |

## B. Account attributes

| Feature | Type | Label | Definition |
| --- | --- | --- | --- |
| `from_entity_type` | categorical | Sender owner type | Owner type of the sender's account: Partnership, Corporation, Sole Proprietorship, Individual, Country or Direct |
| `from_bank_country` | categorical | Sender bank country | Country of the sender's bank, read from the bank name; Crypto for crypto banks, United States for unnumbered US bank names |
| `from_owner_accounts_log` | numeric | Sender owner's accounts (log10) | log10 of the number of accounts held by the sender's owner |
| `from_owner_banks_log` | numeric | Sender owner's banks (log10) | log10 of the number of banks at which the sender's owner holds accounts |
| `to_entity_type` | categorical | Receiver owner type | Owner type of the receiver's account: Partnership, Corporation, Sole Proprietorship, Individual, Country or Direct |
| `to_bank_country` | categorical | Receiver bank country | Country of the receiver's bank, read from the bank name; Crypto for crypto banks, United States for unnumbered US bank names |
| `to_owner_accounts_log` | numeric | Receiver owner's accounts (log10) | log10 of the number of accounts held by the receiver's owner |
| `to_owner_banks_log` | numeric | Receiver owner's banks (log10) | log10 of the number of banks at which the receiver's owner holds accounts |
| `is_cross_border` | flag | Cross-border transfer | 1 if the two banks are in different countries |
| `is_same_owner` | flag | Same owner | 1 if both accounts belong to the same owner, self-transfers included |

## Excluded on purpose

| Left out | Reason |
| --- | --- |
| Day of week | Six training days, so each weekday appears once and would identify a single date; replaced by the weekend flag |
| Calendar date, day of month, time since data start | A hidden clock that points at the simulation tail |
| Account, bank and owner IDs | Memorization across the 147 laundering attempts that span splits; IDs are kept for joins only |
| Lifetime totals | They grow with time |
| Past laundering of an account | Label leakage |
| Target encoding | Not needed; XGBoost handles categories natively |
| Round-amount flags | No signal on the training days |
