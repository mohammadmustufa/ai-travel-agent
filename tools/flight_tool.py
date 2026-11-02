
# Example free API usage
# - AviationStack


# create api key
# https://aviationstack.com/
# pip install requests


import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("AVIATIONSTACK_API_KEY")


def search_flights(dep_iata=None, arr_iata=None, limit=5):
    """Look up live flights, filtered by departure/arrival IATA codes.

    AviationStack is a real-time flight-tracking API: it returns flights
    currently scheduled/active on the given route, not future dated trips
    and not ticket prices.
    """

    url = "http://api.aviationstack.com/v1/flights"

    params = {
        "access_key": API_KEY,
        "limit": limit,
    }

    # Filter by route so we don't get random global flights.
    if dep_iata:
        params["dep_iata"] = dep_iata.strip().upper()
    if arr_iata:
        params["arr_iata"] = arr_iata.strip().upper()

    try:
        response = requests.get(url, params=params, timeout=20)
        data = response.json()
    except requests.RequestException as exc:
        return f"Flight lookup failed (network error): {exc}"
    except ValueError:
        return "Flight lookup failed: invalid response from AviationStack."

    # AviationStack reports errors inside the JSON body, not via HTTP status.
    if isinstance(data, dict) and data.get("error"):
        err = data["error"]
        message = err.get("message") or err.get("info") or str(err)
        return f"Flight lookup error from AviationStack: {message}"

    route = f"{dep_iata or '?'} -> {arr_iata or '?'}"
    results = data.get("data", []) if isinstance(data, dict) else []

    if not results:
        return (
            f"No live flights currently found for {route}. "
            "This route may have no direct flights, or there is no live "
            "flight data for it right now (AviationStack only tracks "
            "current/scheduled flights, not future dated trips or fares)."
        )

    flights = []
    for flight in results[:limit]:
        airline = (flight.get("airline") or {}).get("name") or "Unknown"

        dep = flight.get("departure") or {}
        arr = flight.get("arrival") or {}
        flight_info = flight.get("flight") or {}

        departure = dep.get("airport") or dep.get("iata") or "Unknown"
        arrival = arr.get("airport") or arr.get("iata") or "Unknown"
        flight_number = flight_info.get("iata") or flight_info.get("number") or "N/A"
        dep_time = dep.get("scheduled") or "N/A"
        status = flight.get("flight_status") or "Unknown"
        flight_date = flight.get("flight_date") or "N/A"

        flights.append(
            f"""
Airline: {airline}
Flight: {flight_number}
Date: {flight_date}
Departure: {departure} (scheduled: {dep_time})
Arrival: {arrival}
Status: {status}
"""
        )

    return "\n".join(flights)
