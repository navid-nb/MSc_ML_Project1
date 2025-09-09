-- Authentification for Corporate Bonds data
-- Source: FISD (Fixed Income Securities Database) via WRDS
SELECT
    exchange,
    issuer_id,
    ticker as "tic"
FROM fisd.fisd_exchange_listing
