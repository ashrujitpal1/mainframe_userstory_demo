from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Uncomment the following line to use an example of a custom tool
# from mainframe_userstory_demo.tools.custom_tool import MyCustomTool

# Check our tools documentations for more information on how to use them
# from crewai_tools import SerperDevTool

@CrewBase
class MainframeJavaCodeDemoCrew():
	"""MainframeJavaCodeDemo crew"""

	llm = LLM(model="ollama/llama3.2", base_url="http://localhost:11434")

	@agent
	def javacodewriter(self) -> Agent:
		return Agent(
			config=self.agents_config['javacodewriter'],
			# tools=[MyCustomTool()], # Example of custom tool, loaded on the beginning of file
			verbose=True,
			llm=self.llm,
			allow_delegation=False
		)
	
	# # User Story reviewer
	# @agent
	# def userstoryreviewer(self) -> Agent:
	# 	return Agent(
	# 		config=self.agents_config['userstoryreviewer'],
	# 		# tools=[MyCustomTool()], # Example of custom tool, loaded on the beginning of file
	# 		verbose=True,
	# 		llm=self.llm,
	# 		allow_delegation=False
	#	)

	@task
	def javacodewriter_task(self) -> Task:
		return Task(
			config=self.tasks_config['javacodewriter_task'],
		)
	
	# @task
	# def userstoryreviewer_task(self) -> Task:
	# 	return Task(
	# 		config=self.tasks_config['userstoryreviewer_task'],
	# 		agent=self.userstoryreviewer()
	# 	)

	@crew
	def crew(self) -> Crew:
		"""Creates the MainframeJavaCodeDemo crew"""
		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)