import { useState, useEffect, useRef, useCallback } from 'react';
import { getTask, type TaskRecord } from '../api/client';

/**
 * Hook that polls a background task until it completes or fails.
 * Returns { task, startPolling, reset }.
 */
export function useTaskPoller(intervalMs = 2000) {
  const [task, setTask] = useState<TaskRecord | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const taskIdRef = useRef<string | null>(null);

  const stopPolling = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const poll = useCallback(async (id: string) => {
    try {
      const t = await getTask(id);
      setTask(t);
      if (t.status === 'Completed' || t.status === 'Failed') {
        stopPolling();
      }
    } catch {
      // keep polling on transient errors
    }
  }, [stopPolling]);

  const startPolling = useCallback((taskId: string) => {
    stopPolling();
    taskIdRef.current = taskId;
    setTask({ id: taskId, task_type: '', status: 'Running', created_at: '', updated_at: '', progress: null, result: null, error: null });
    poll(taskId);
    timerRef.current = setInterval(() => poll(taskId), intervalMs);
  }, [intervalMs, poll, stopPolling]);

  const reset = useCallback(() => {
    stopPolling();
    taskIdRef.current = null;
    setTask(null);
  }, [stopPolling]);

  // Cleanup on unmount
  useEffect(() => stopPolling, [stopPolling]);

  // On mount, check if there's a stored task ID in sessionStorage
  useEffect(() => {
    // Intentionally no auto-restore here — components handle their own persistence
  }, []);

  return { task, startPolling, reset };
}
