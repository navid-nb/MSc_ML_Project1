-- IBES Actuals (earnings announcements and realized values)
SELECT ticker,                -- IBES Ticker Symbol
       cusip,                 -- CUSIP/SEDOL (security identifier)
       oftic,                 -- Official Ticker Symbol
       cname,                 -- Company Name

       -- Event timing
       anndats,               -- Announce date (when actual EPS was released)
       anntims,               -- Announce time (can be 'before open', 'after close')
       actdats,               -- Activation date (when IBES loaded the record)
       acttims,               -- Activation time (system-level, not always aligned)

       -- Fundamental context
       pends,                 -- Fiscal period end date
       pdicity,               -- Periodicity (Q = quarterly, A = annual)
       measure AS act_measure,-- Measure type (EPS, CFPS, DPS, etc.)
       value   AS act_value,  -- Reported actual value
       curr_act,              -- Currency of reported actual
       usfirm                 -- Flag: 1 if US firm, 0 if INT
FROM ibes.actu_epsus
WHERE anndats < %(end)s::date
  AND measure = 'EPS'   -- focus on EPS actuals (because we are forcasting future returns)
  AND usfirm = 1; -- keep only U.S. firms (on nasdaq + nyse)