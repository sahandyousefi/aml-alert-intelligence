# Feature dictionary

Generated from `src/aml_alert_intelligence/features/registry.py` by `scripts/make_feature_docs.py`. Do not edit by hand.

46 features. Flags are 0 or 1. No feature uses the label, a calendar date, or an account identifier, and none is fitted on anything but training days.

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

## C. Account behavior

Every window ends one minute before the transfer, so entries in the same minute are left out. Self-transfers get features but never count as history.

| Feature | Type | Label | Definition |
| --- | --- | --- | --- |
| `from_entries_1h` | numeric | Sender: transfers in the past hour | Transfers sent or received by the sender's account in the 60 minutes before |
| `from_in_24h` | numeric | Sender: transfers received, past 24 hours | Transfers received by the sender's account in the 24 hours before |
| `from_out_24h` | numeric | Sender: transfers sent, past 24 hours | Transfers sent by the sender's account in the 24 hours before |
| `from_in_usd_24h_log` | numeric | Sender: US dollars received, past 24 hours (log10) | log10(1 + US dollars received by the sender's account in the 24 hours before) |
| `from_out_usd_24h_log` | numeric | Sender: US dollars sent, past 24 hours (log10) | log10(1 + US dollars sent by the sender's account in the 24 hours before) |
| `from_in_72h` | numeric | Sender: transfers received, past 72 hours | Transfers received by the sender's account in the 72 hours before |
| `from_out_72h` | numeric | Sender: transfers sent, past 72 hours | Transfers sent by the sender's account in the 72 hours before |
| `from_in_usd_72h_log` | numeric | Sender: US dollars received, past 72 hours (log10) | log10(1 + US dollars received by the sender's account in the 72 hours before) |
| `from_out_usd_72h_log` | numeric | Sender: US dollars sent, past 72 hours (log10) | log10(1 + US dollars sent by the sender's account in the 72 hours before) |
| `from_new_payers_24h` | numeric | Sender: new payers, past 24 hours | Accounts that paid the sender's account in the past 24 hours with no payment between them in the 72 hours before |
| `from_new_payees_24h` | numeric | Sender: new payees, past 24 hours | Accounts that the sender's account paid in the past 24 hours with no payment between them in the 72 hours before |
| `from_minutes_since_last` | numeric | Sender: minutes since previous transfer | Minutes since the sender's account last sent or received a transfer, capped at 72 hours (4,320), which also stands for no history |
| `to_entries_1h` | numeric | Receiver: transfers in the past hour | Transfers sent or received by the receiver's account in the 60 minutes before |
| `to_in_24h` | numeric | Receiver: transfers received, past 24 hours | Transfers received by the receiver's account in the 24 hours before |
| `to_out_24h` | numeric | Receiver: transfers sent, past 24 hours | Transfers sent by the receiver's account in the 24 hours before |
| `to_in_usd_24h_log` | numeric | Receiver: US dollars received, past 24 hours (log10) | log10(1 + US dollars received by the receiver's account in the 24 hours before) |
| `to_out_usd_24h_log` | numeric | Receiver: US dollars sent, past 24 hours (log10) | log10(1 + US dollars sent by the receiver's account in the 24 hours before) |
| `to_in_72h` | numeric | Receiver: transfers received, past 72 hours | Transfers received by the receiver's account in the 72 hours before |
| `to_out_72h` | numeric | Receiver: transfers sent, past 72 hours | Transfers sent by the receiver's account in the 72 hours before |
| `to_in_usd_72h_log` | numeric | Receiver: US dollars received, past 72 hours (log10) | log10(1 + US dollars received by the receiver's account in the 72 hours before) |
| `to_out_usd_72h_log` | numeric | Receiver: US dollars sent, past 72 hours (log10) | log10(1 + US dollars sent by the receiver's account in the 72 hours before) |
| `to_new_payers_24h` | numeric | Receiver: new payers, past 24 hours | Accounts that paid the receiver's account in the past 24 hours with no payment between them in the 72 hours before |
| `to_new_payees_24h` | numeric | Receiver: new payees, past 24 hours | Accounts that the receiver's account paid in the past 24 hours with no payment between them in the 72 hours before |
| `to_minutes_since_last` | numeric | Receiver: minutes since previous transfer | Minutes since the receiver's account last sent or received a transfer, capped at 72 hours (4,320), which also stands for no history |
| `pair_prior_72h` | numeric | Earlier transfers between the pair, past 72 hours | Earlier transfers from this sender to this receiver in the 72 hours before |
| `reverse_prior_72h` | numeric | Earlier transfers back, past 72 hours | Earlier transfers from this receiver to this sender in the 72 hours before; money flowing back is a sign of a cycle |
| `sender_passthrough_24h` | numeric | Payment covered by the sender's recent inflow | min(amount, US dollars the sender received in the 24 hours before) / amount; 1 means the sender received at least this much in the past day |

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
