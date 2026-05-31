"""Proactive, always-on behavior: Aria initiates on triggers/intervals.

The turn pipeline stays reactive; this package only decides *when* Aria should speak
and routes a prompt into the pipeline via NursePipeline.engage().
"""
