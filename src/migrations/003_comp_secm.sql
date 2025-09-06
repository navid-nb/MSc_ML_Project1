-- Compustat security monthly descriptor (only fields we need)
SELECT
  gvkey,
  tic,
  cusip,
  datadate
FROM comp.secm
WHERE datadate < %(end)s::date;  -- we only need snapshots up to end
