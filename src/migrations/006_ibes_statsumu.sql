-- IBES summary (keep columns we actually use)
SELECT
  ticker,
  statpers,
  measure,
  fiscalp,
  fpi,
  estflag,
  curcode
FROM ibes.statsumu_epsus
WHERE statpers < %(end)s::date;
