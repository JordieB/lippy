import os
from pydantic import BaseModel, Field
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.agents import Tool
from langchain.chat_models import ChatOpenAI
from gcalwrapper import GoogleCalendar


class QuestionInput(BaseModel):
    """Input model for the NextMeetingTool tools."""
    question: str = Field()


class NextMeetingTool:
    """
    A class to represent the NextMeetingTool.

    Attributes:
        google_calendar (GoogleCalendar): GoogleCalendar object to interact with the calendar API.
        llm (ChatOpenAI): The language model object.
        template (PromptTemplate): The template for the prompt.
        next_meeting_chain (LLMChain): The chain object for running the language model.
        tools (list[Tool]): A list of tools for getting calendar events and next event.
    """

    def __init__(self, service_file_path: str, calendar_id: str):
        """
        Initializes the NextMeetingTool with the specified service_file_path and calendar_id.

        Args:
            service_file_path (str): The path to the service file for Google Calendar API.
            calendar_id (str): The Google Calendar ID to access.
        """
        self.google_calendar = GoogleCalendar(service_file_path, calendar_id)
        self.llm = ChatOpenAI(temperature=0)
        self.template = PromptTemplate(
            input_variables=['event_context', 'current_ts'],
            template="===\nUpcoming Events: {event_context}\n===\nQ: The current time is {current_ts}. "
                     "What and when is my next meeting?\nA: "
        )
        self.next_meeting_chain = LLMChain(llm=self.llm, prompt=self.template)

        self.tools = [
            Tool(
                name="Get Google Calendar Events",
                func=self.google_calendar.get_calendar_events,
                description="useful for when you need to grab the next n calendar events"
            ),
            Tool(
                name="Get Next Event from Google Calendar",
                func=self.next_meeting_chain.run,
                description='asks an LLM for your next event',
                args_schema=QuestionInput
            )
        ]

    def get_next_meeting(self) -> str:
        """
        Get the next meeting using Google Calendar and the LLM chain.

        Returns:
            str: The next meeting as a formatted string.
        """
        events = self.google_calendar.get_calendar_events()
        events = [f"{event['summary']} @ {event['start']['dateTime']}" for event in events]
        events = '\n'.join(events)
        current_ts = self.google_calendar.current_utc_timestamp()

        inputs = {'event_context': events, 'current_ts': current_ts}
        return self.next_meeting_chain(inputs=inputs, return_only_outputs=True)['text']


if __name__ == '__main__':
    service_file_path = '/work/cal_dump.pkl'
    calendar_id = os.environ['GOOGLE_CAL_ID']

    next_meeting_tool = NextMeetingTool(service_file_path, calendar_id)
    next_meeting = next_meeting_tool.get_next_meeting()
    print("Next meeting:", next_meeting)
