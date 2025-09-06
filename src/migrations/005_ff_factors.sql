SELECT date, mktrf, smb, hml, rf, umd
FROM ff.factors_daily
WHERE date BETWEEN %(start)s AND %(end)s;