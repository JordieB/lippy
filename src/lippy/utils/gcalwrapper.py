import datetime
import os
import pickle
from googleapiclient.errors import HttpError


class GoogleCalendar:
    def __init__(self, service_file, calendar_id):
        self.service = self.load_service(service_file)
        self.calendar_id = calendar_id

    @staticmethod
    def load_service(service_file):
        """
        Load the authenticated calendar service object from a file.

        :param service_file: The path to the file containing the pickled service object.
        :return: An authenticated calendar service object.
        """
        try:
            with open(service_file, 'rb') as f:
                service = pickle.load(f)
            return service
        except FileNotFoundError:
            print("Service file not found.")
            return None

    def get_calendar_events(self, max_res=10):
        """
        Retrieve and print upcoming events from the specified Google Calendar.
        """
        if not self.service:
            print("Could not obtain calendar service.")
            return

        try:
            # Get the current time in UTC format
            now = datetime.datetime.utcnow().isoformat() + 'Z'

            # Query the Google Calendar API for upcoming events
            events_result = self.service.events().list(
                calendarId=self.calendar_id,
                timeMin=now,
                maxResults=max_res,
                singleEvents=True,
                orderBy='startTime'
            ).execute()

            # Extract the events from the API response
            events = events_result.get('items', [])

            if not events:
                print('No upcoming events found.')
            else:
                return events

        except HttpError as error:
            print(f"An error occurred: {error}")
            return None

    def current_utc_timestamp(self):
        """
        Return the current timestamp in UTC format.

        :return: A string representing the current UTC timestamp in ISO 8601 format.
        """
        # Get the current time in UTC
        now = datetime.datetime.utcnow()

        # Format the time as an ISO 8601 string with 'Z' indicating UTC
        utc_timestamp = now.isoformat() + 'Z'

        return utc_timestamp


if __name__ == '__main__':
    service_file_path = '/work/cal_dump.pkl'
    calendar_id = os.environ['GOOGLE_CAL_ID']
    google_calendar = GoogleCalendar(service_file_path, calendar_id)
    print(google_calendar.get_calendar_events())
    print(google_calendar.current_utc_timestamp())
