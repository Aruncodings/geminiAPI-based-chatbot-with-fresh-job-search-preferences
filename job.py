
import time
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import List, Optional
import requests

logger = logging.getLogger("EnhancedRealTimeJobSearch")
logger.setLevel(logging.INFO)

JSEARCH_BASE_URL = "https://jsearch.p.rapidapi.com/search"
DEFAULT_RAPIDAPI_HOST = "jsearch.p.rapidapi.com"

class RealTimeJobListing:
    def __init__(self,
                 title: str,
                 company: str,
                 location: str,
                 description: str,
                 url: str,
                 source: str = "jsearch",
                 salary: Optional[str] = None,
                 job_type: Optional[str] = None,
                 posted_date: Optional[str] = None,
                 company_url: Optional[str] = None,
                 job_id: Optional[str] = None,
                 posted_timestamp: Optional[datetime] = None,
                 is_fresh: bool = False,
                 freshness_score: int = 0):
        self.title = title or ""
        self.company = company or ""
        self.location = location or ""
        self.description = description or ""
        self.url = url or ""
        self.source = source or "jsearch"
        self.salary = salary or ""
        self.job_type = job_type or ""
        self.posted_date = posted_date or ""
        self.company_url = company_url or ""
        self.job_id = job_id or (self._make_id())
        self.posted_timestamp = posted_timestamp or datetime.utcnow()
        self.is_fresh = bool(is_fresh)
        self.freshness_score = int(freshness_score or 0)

    def _make_id(self):
        try:
            key = f"{self.company}-{self.title}-{hash(self.url)}"
            return key
        except Exception:
            return f"job-{hash(self.url)}"

    def to_dict(self):
        hours_old = 999
        try:
            hours_old = (datetime.utcnow() - (self.posted_timestamp or datetime.utcnow())).total_seconds() / 3600.0
        except Exception:
            hours_old = 999

        return {
            "title": self.title,
            "company": self.company,
            "location": self.location,
            "description": self.description,
            "description_html": f"<p>{(self.description or '').replace(chr(10), '<br/>')}</p>",
            "url": self.url,
            "company_url": self.company_url,
            "source": self.source,
            "salary": self.salary,
            "job_type": self.job_type,
            "posted_date": self.posted_date,
            "posted_timestamp": (self.posted_timestamp.isoformat() if isinstance(self.posted_timestamp, datetime) else None),
            "job_id": self.job_id,
            "is_fresh": bool(self.is_fresh),
            "freshness_score": int(self.freshness_score),
            "hours_old": hours_old
        }

class EnhancedRealTimeJobSearch:
    def __init__(self, rapidapi_key: str = "", rapidapi_host: str = DEFAULT_RAPIDAPI_HOST, db_path: str = "users.db"):
        """
        rapidapi_key: your RapidAPI key for JSearch
        rapidapi_host: normally 'jsearch.p.rapidapi.com'
        db_path: path to the sqlite db used by main.py (defaults to users.db)
        """
        self.rapidapi_key = rapidapi_key or ""
        self.rapidapi_host = rapidapi_host or DEFAULT_RAPIDAPI_HOST
        self.db_path = db_path

        self.headers = {
            "X-RapidAPI-Key": self.rapidapi_key,
            "X-RapidAPI-Host": self.rapidapi_host,
            "Accept": "application/json"
        }

    # Public search function
    def search_jobs(self, job_title: str, location: str = "", max_age_hours: int = 168, limit: int = 20) -> List[RealTimeJobListing]:
        """
        Search using JSearch RapidAPI. Returns list of RealTimeJobListing objects.
        - job_title: required
        - location: optional
        - max_age_hours: freshness window (e.g., 24, 72, 168)
        - limit: max results to return
        """
        if not job_title or not job_title.strip():
            return []

        query_text = f"{job_title} {location}".strip()

        params = {
            "query": query_text,
            "page": "1",
            "num_pages": "1"
        }

        # Retry/backoff parameters
        max_attempts = 3
        backoff_seconds = 1.0
        resp_json = None

        # If API key is empty, return empty list early (avoid network calls)
        if not self.rapidapi_key:
            logger.warning("JSearch API key not configured. Returning empty result.")
            return []

        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(f"[JSearch] attempt {attempt} query={query_text}")
                resp = requests.get(JSEARCH_BASE_URL, headers=self.headers, params=params, timeout=12)
                if resp.status_code == 429:
                    logger.warning("[JSearch] rate limited (429). Backing off.")
                    time.sleep(backoff_seconds * attempt)
                    continue

                resp.raise_for_status()
                resp_json = resp.json()
                break
            except requests.RequestException as e:
                logger.warning(f"[JSearch] request exception (attempt {attempt}): {e}")
                if attempt < max_attempts:
                    time.sleep(backoff_seconds * attempt)
                else:
                    logger.error("[JSearch] all attempts failed.")
                    resp_json = None

        if not resp_json:
            return []

        data_list = resp_json.get("data") or resp_json.get("results") or []
        if not isinstance(data_list, list):
            logger.warning("[JSearch] unexpected response shape")
            return []

        jobs: List[RealTimeJobListing] = []
        seen_ids = set()

        for raw in data_list[:limit]:
            try:
                title = raw.get("job_title") or raw.get("title") or raw.get("position") or ""
                company = raw.get("employer_name") or raw.get("company_name") or raw.get("company") or ""
                city = raw.get("job_city") or raw.get("location") or raw.get("job_location") or ""
                country = raw.get("job_country") or raw.get("country") or ""
                location_text = ", ".join(filter(None, [city, country])).strip() or city or country or ""
                description = raw.get("job_description") or raw.get("description") or raw.get("snippet") or ""
                url = raw.get("job_apply_link") or raw.get("url") or raw.get("apply_link") or ""
                salary = raw.get("job_salary") or raw.get("salary") or raw.get("job_salary_currency") or ""
                job_type = raw.get("job_employment_type") or raw.get("employment_type") or raw.get("job_type") or ""
                posted_iso = raw.get("job_posted_at_datetime_utc") or raw.get("posted_at") or raw.get("job_posted_at")
                company_url = raw.get("employer_website") or raw.get("company_website") or raw.get("company_url") or ""
                job_id = raw.get("job_id") or raw.get("id") or (company + "-" + title + "-" + str(hash(url)))

                posted_ts = None
                if posted_iso:
                    posted_ts = self._parse_iso_to_dt(posted_iso)

                # compute hours_old and freshness
                hours_old = 999.0
                if posted_ts:
                    hours_old = (datetime.utcnow() - posted_ts).total_seconds() / 3600.0

                is_fresh = hours_old <= max_age_hours
                freshness_score = self._calc_freshness_score(hours_old, max_age_hours) if is_fresh else 0

                listing = RealTimeJobListing(
                    title=title.strip(),
                    company=company.strip(),
                    location=location_text.strip() or location.strip(),
                    description=description.strip(),
                    url=url.strip(),
                    source="jsearch",
                    salary=str(salary).strip(),
                    job_type=str(job_type).strip(),
                    posted_date=raw.get("job_posted_at") or raw.get("posted_at") or "",
                    company_url=company_url.strip(),
                    job_id=str(job_id),
                    posted_timestamp=posted_ts or (datetime.utcnow() - timedelta(hours=hours_old) if hours_old < 999 else datetime.utcnow()),
                    is_fresh=is_fresh,
                    freshness_score=int(freshness_score)
                )

                dedupe_key = (listing.job_id or "").strip() or (listing.url or "").strip()
                if not dedupe_key:
                    dedupe_key = f"{listing.title}-{listing.company}-{hash(listing.url)}"

                if dedupe_key in seen_ids:
                    continue
                seen_ids.add(dedupe_key)

                jobs.append(listing)
            except Exception as e:
                logger.warning(f"[JSearch] failed to parse job entry: {e}")
                continue

        return jobs

    # DB helpers
    def _get_db_conn(self):
        conn = sqlite3.connect(self.db_path, timeout=20.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def save_job_search(self, user_id: int, query: str, location: str, results_count: int):
        try:
            conn = self._get_db_conn()
            c = conn.cursor()
            c.execute('''
                INSERT INTO job_searches (user_id, query, location, results_count, search_time, max_age_hours)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
            ''', (user_id, query, location, results_count, None))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to save job search history: {e}")

    def save_job(self, user_id: int, job: RealTimeJobListing) -> bool:
        try:
            conn = self._get_db_conn()
            c = conn.cursor()
            c.execute('''
                INSERT INTO saved_jobs (
                    user_id, job_id, title, company, location, description, url, company_url,
                    source, salary, job_type, posted_date, posted_timestamp, freshness_score, is_fresh
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                job.job_id,
                job.title,
                job.company,
                job.location,
                job.description,
                job.url,
                job.company_url or "",
                job.source or "",
                job.salary or "",
                job.job_type or "",
                job.posted_date or "",
                (job.posted_timestamp.isoformat() if isinstance(job.posted_timestamp, datetime) else None),
                int(job.freshness_score or 0),
                int(bool(job.is_fresh))
            ))
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError as e:
            logger.info(f"Job already saved or integrity error: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to save job to DB: {e}")
            return False

    def get_saved_jobs(self, user_id: int, limit: Optional[int] = None):
        try:
            conn = self._get_db_conn()
            c = conn.cursor()
            sql = "SELECT * FROM saved_jobs WHERE user_id = ? ORDER BY saved_at DESC"
            params = [user_id]
            if limit:
                sql += " LIMIT ?"
                params.append(int(limit))
            c.execute(sql, tuple(params))
            rows = [dict(row) for row in c.fetchall()]
            conn.close()
            return rows
        except Exception as e:
            logger.error(f"Failed to query saved jobs: {e}")
            return []

    def get_search_history(self, user_id: int, limit: int = 10):
        try:
            conn = self._get_db_conn()
            c = conn.cursor()
            c.execute('''
                SELECT id, user_id, query, location, results_count, search_time, search_metadata, max_age_hours
                FROM job_searches
                WHERE user_id = ?
                ORDER BY search_time DESC
                LIMIT ?
            ''', (user_id, min(int(limit), 100)))
            rows = [dict(row) for row in c.fetchall()]
            conn.close()
            return rows
        except Exception as e:
            logger.error(f"Failed to fetch search history: {e}")
            return []

    # Utilities
    @staticmethod
    def _parse_iso_to_dt(iso_str: str) -> Optional[datetime]:
        if not iso_str:
            return None
        fmts = [
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d"
        ]
        for f in fmts:
            try:
                return datetime.strptime(iso_str, f)
            except Exception:
                continue
        try:
            return datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        except Exception:
            return None

    @staticmethod
    def _calc_freshness_score(hours_old: float, max_age_hours: float) -> int:
        try:
            hours_old = float(hours_old or 0)
            max_age_hours = float(max_age_hours or 1)
            if hours_old <= 0:
                return 100
            if hours_old >= max_age_hours:
                return 0
            score = int(max(0, min(100, round((1 - (hours_old / max_age_hours)) * 100))))
            return score
        except Exception:
            return 0