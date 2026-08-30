"""
Import-only. Single source of truth for the calendar / season conventions.

  MONTH_SEASON         month NAME  -> season   (used by demand_profile_model / optimisation model)
  MONTH_SEASON_BY_NUM  month 1..12 -> season   (used by the api_* pullers, which work in datetimes)
  MONTHS_ORDER         January .. December
  MONTH_DAYS           month name -> days (non-leap basis; the model year is 365 days)
  SEASONS              season order used by the api_* workbook columns
  DAY_TYPES            representative day types (weekday / weekend)
  HH_PER_DAY           half-hour slots per day
  HOURS_PER_DAY        hours per day
"""

MONTH_SEASON = {
    "January": "Winter",  "February": "Winter",  "March":     "Spring",
    "April":   "Spring",  "May":      "Spring",  "June":      "Summer",
    "July":    "Summer",  "August":   "Summer",  "September": "Autumn",
    "October": "Autumn",  "November": "Autumn",  "December":  "Winter",
}
MONTH_DAYS = {
    "January":   31, "February": 28, "March":     31, "April":   30,
    "May":       31, "June":     30, "July":      31, "August":  31,
    "September": 30, "October":  31, "November":  30, "December": 31,
}
MONTHS_ORDER = list(MONTH_DAYS.keys())

# Same mapping keyed by calendar month number, for callers working in datetimes.
MONTH_SEASON_BY_NUM = {i: MONTH_SEASON[m] for i, m in enumerate(MONTHS_ORDER, 1)}

SEASONS       = ["Winter", "Spring", "Summer", "Autumn"]
DAY_TYPES     = ["WD", "WE"]
HH_PER_DAY    = 48
HOURS_PER_DAY = 24

__all__ = ["MONTH_SEASON", "MONTH_SEASON_BY_NUM", "MONTHS_ORDER", "MONTH_DAYS",
           "SEASONS", "DAY_TYPES", "HH_PER_DAY", "HOURS_PER_DAY"]
