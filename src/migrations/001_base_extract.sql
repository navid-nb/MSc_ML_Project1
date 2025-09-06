-- CRSP daily (bounded by date)
SELECT
  permno,
  "date",
  ret,
  vol,
  prc,
  shrout
FROM crsp.dsf
WHERE "date" >= %(start)s::date
  AND "date" <  %(end)s::date;