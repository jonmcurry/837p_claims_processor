import React, { useState, useCallback, useMemo } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  MenuItem,
  Grid,
  Chip,
  IconButton,
  Tooltip,
  Alert,
  Snackbar,
  FormControl,
  InputLabel,
  Select,
  SelectChangeEvent,
} from '@mui/material';
import {
  DataGrid,
  GridColDef,
  GridRowParams,
  GridToolbarContainer,
  GridToolbarExport,
  GridToolbarFilterButton,
  GridToolbarColumnsButton,
  GridActionsCellItem,
  GridRowId,
} from '@mui/x-data-grid';
import {
  Visibility as ViewIcon,
  Edit as EditIcon,
  Assignment as AssignIcon,
  CheckCircle as ResolveIcon,
  Refresh as RefreshIcon,
  Download as ExportIcon,
  FilterList as FilterIcon,
} from '@mui/icons-material';
import { DatePicker } from '@mui/x-date-pickers/DatePicker';
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider';
import { AdapterDateFns } from '@mui/x-date-pickers/AdapterDateFns';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { useSnackbar } from 'notistack';
import { format, parseISO } from 'date-fns';

import { useAuth } from '../../hooks/useAuth';
import { useWebSocket } from '../../hooks/useWebSocket';
import { failedClaimsApi, FailedClaim, ClaimFilters, ResolutionRequest } from '../../services/api';
import LoadingSpinner from '../../components/LoadingSpinner/LoadingSpinner';
import FailedClaimDetailModal from '../../components/FailedClaimDetailModal/FailedClaimDetailModal';
import ClaimResolutionModal from '../../components/ClaimResolutionModal/ClaimResolutionModal';

interface FailedClaimsState {
  filters: ClaimFilters;
  selectedClaims: GridRowId[];
  detailModalOpen: boolean;
  resolutionModalOpen: boolean;
  assignModalOpen: boolean;
  selectedClaim: FailedClaim | null;
  snackbarOpen: boolean;
  snackbarMessage: string;
  snackbarSeverity: 'success' | 'error' | 'warning' | 'info';
}

const initialFilters: ClaimFilters = {
  facilityId: '',
  failureCategory: '',
  resolutionStatus: '',
  dateRange: {
    startDate: null,
    endDate: null,
  },
  priority: '',
  assignedTo: '',
};

const failureCategories = [
  'validation_error',
  'missing_data',
  'duplicate_claim',
  'invalid_facility',
  'invalid_provider',
  'invalid_procedure',
  'invalid_diagnosis',
  'date_range_error',
  'financial_error',
  'ml_rejection',
  'system_error',
];

const resolutionStatuses = [
  'pending',
  'in_progress',
  'resolved',
  'rejected',
  'escalated',
];

const priorities = ['low', 'medium', 'high', 'critical'];

const FailedClaims: React.FC = () => {
  const { user } = useAuth();
  const { enqueueSnackbar } = useSnackbar();
  const queryClient = useQueryClient();

  const [state, setState] = useState<FailedClaimsState>({
    filters: initialFilters,
    selectedClaims: [],
    detailModalOpen: false,
    resolutionModalOpen: false,
    assignModalOpen: false,
    selectedClaim: null,
    snackbarOpen: false,
    snackbarMessage: '',
    snackbarSeverity: 'info',
  });

  // WebSocket for real-time updates
  useWebSocket('failed_claims', (data) => {
    queryClient.invalidateQueries(['failed-claims']);
    
    if (data.type === 'claim_failed') {
      enqueueSnackbar(`New failed claim: ${data.claim_id}`, { variant: 'warning' });
    } else if (data.type === 'claim_resolved') {
      enqueueSnackbar(`Claim resolved: ${data.claim_id}`, { variant: 'success' });
    }
  });

  // Fetch failed claims
  const {
    data: failedClaimsData,
    isLoading,
    error,
    refetch,
  } = useQuery(
    ['failed-claims', state.filters],
    () => failedClaimsApi.getFailedClaims(state.filters),
    {
      keepPreviousData: true,
      refetchInterval: 30000, // Refetch every 30 seconds
    }
  );

  // Mutations
  const resolveMutation = useMutation(
    (data: ResolutionRequest) => failedClaimsApi.resolveClaim(data.claimId, data),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['failed-claims']);
        enqueueSnackbar('Claim resolved successfully', { variant: 'success' });
        setState(prev => ({ ...prev, resolutionModalOpen: false, selectedClaim: null }));
      },
      onError: (error: any) => {
        enqueueSnackbar(`Resolution failed: ${error.message}`, { variant: 'error' });
      },
    }
  );

  const assignMutation = useMutation(
    ({ claimIds, assignedTo }: { claimIds: string[]; assignedTo: string }) =>
      failedClaimsApi.assignClaims(claimIds, assignedTo),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['failed-claims']);
        enqueueSnackbar('Claims assigned successfully', { variant: 'success' });
        setState(prev => ({ ...prev, assignModalOpen: false, selectedClaims: [] }));
      },
      onError: (error: any) => {
        enqueueSnackbar(`Assignment failed: ${error.message}`, { variant: 'error' });
      },
    }
  );

  // Event handlers
  const handleFilterChange = useCallback((field: keyof ClaimFilters, value: any) => {
    setState(prev => ({
      ...prev,
      filters: {
        ...prev.filters,
        [field]: value,
      },
    }));
  }, []);

  const handleViewClaim = useCallback((claim: FailedClaim) => {
    setState(prev => ({
      ...prev,
      selectedClaim: claim,
      detailModalOpen: true,
    }));
  }, []);

  const handleResolveClaim = useCallback((claim: FailedClaim) => {
    setState(prev => ({
      ...prev,
      selectedClaim: claim,
      resolutionModalOpen: true,
    }));
  }, []);

  const handleAssignClaims = useCallback(() => {
    if (state.selectedClaims.length === 0) {
      enqueueSnackbar('Please select claims to assign', { variant: 'warning' });
      return;
    }
    setState(prev => ({ ...prev, assignModalOpen: true }));
  }, [state.selectedClaims.length, enqueueSnackbar]);

  const handleBulkExport = useCallback(() => {
    if (state.selectedClaims.length === 0) {
      enqueueSnackbar('Please select claims to export', { variant: 'warning' });
      return;
    }
    
    // Export selected claims
    failedClaimsApi.exportClaims(state.selectedClaims as string[])
      .then(() => {
        enqueueSnackbar('Export completed', { variant: 'success' });
      })
      .catch((error) => {
        enqueueSnackbar(`Export failed: ${error.message}`, { variant: 'error' });
      });
  }, [state.selectedClaims, enqueueSnackbar]);

  const handleClearFilters = useCallback(() => {
    setState(prev => ({ ...prev, filters: initialFilters }));
  }, []);

  // Column definitions
  const columns: GridColDef[] = useMemo(
    () => [
      {
        field: 'claimReference',
        headerName: 'Claim ID',
        width: 150,
        renderCell: (params) => (
          <Tooltip title="Click to view details">
            <Button
              variant="text"
              size="small"
              onClick={() => handleViewClaim(params.row)}
              sx={{ textTransform: 'none' }}
            >
              {params.value}
            </Button>
          </Tooltip>
        ),
      },
      {
        field: 'facilityId',
        headerName: 'Facility',
        width: 120,
      },
      {
        field: 'failureCategory',
        headerName: 'Category',
        width: 150,
        renderCell: (params) => (
          <Chip
            label={params.value?.replace('_', ' ').toUpperCase()}
            size="small"
            color={
              params.value === 'system_error' ? 'error' :
              params.value === 'validation_error' ? 'warning' :
              'default'
            }
          />
        ),
      },
      {
        field: 'failureReason',
        headerName: 'Reason',
        width: 250,
        renderCell: (params) => (
          <Tooltip title={params.value}>
            <span
              style={{
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
              }}
            >
              {params.value}
            </span>
          </Tooltip>
        ),
      },
      {
        field: 'chargeAmount',
        headerName: 'Amount',
        width: 120,
        type: 'number',
        valueFormatter: (params) => `$${params.value?.toLocaleString() || '0'}`,
      },
      {
        field: 'resolutionStatus',
        headerName: 'Status',
        width: 130,
        renderCell: (params) => (
          <Chip
            label={params.value?.toUpperCase()}
            size="small"
            color={
              params.value === 'resolved' ? 'success' :
              params.value === 'in_progress' ? 'primary' :
              params.value === 'escalated' ? 'error' :
              'default'
            }
          />
        ),
      },
      {
        field: 'assignedTo',
        headerName: 'Assigned To',
        width: 150,
        renderCell: (params) => params.value || 'Unassigned',
      },
      {
        field: 'failedAt',
        headerName: 'Failed At',
        width: 150,
        type: 'dateTime',
        valueFormatter: (params) =>
          params.value ? format(parseISO(params.value), 'MMM dd, HH:mm') : '',
      },
      {
        field: 'actions',
        type: 'actions',
        headerName: 'Actions',
        width: 150,
        getActions: (params: GridRowParams) => [
          <GridActionsCellItem
            icon={<ViewIcon />}
            label="View Details"
            onClick={() => handleViewClaim(params.row)}
          />,
          <GridActionsCellItem
            icon={<EditIcon />}
            label="Resolve"
            onClick={() => handleResolveClaim(params.row)}
            disabled={params.row.resolutionStatus === 'resolved'}
          />,
        ],
      },
    ],
    [handleViewClaim, handleResolveClaim]
  );

  // Custom toolbar
  const CustomToolbar: React.FC = () => (
    <GridToolbarContainer>
      <GridToolbarFilterButton />
      <GridToolbarColumnsButton />
      <GridToolbarExport />
      <Button
        startIcon={<AssignIcon />}
        onClick={handleAssignClaims}
        disabled={state.selectedClaims.length === 0}
        size="small"
      >
        Assign Selected
      </Button>
      <Button
        startIcon={<ExportIcon />}
        onClick={handleBulkExport}
        disabled={state.selectedClaims.length === 0}
        size="small"
      >
        Export Selected
      </Button>
      <Button
        startIcon={<RefreshIcon />}
        onClick={() => refetch()}
        size="small"
      >
        Refresh
      </Button>
    </GridToolbarContainer>
  );

  if (error) {
    return (
      <Box p={3}>
        <Alert severity="error">
          Error loading failed claims: {(error as Error).message}
        </Alert>
      </Box>
    );
  }

  return (
    <LocalizationProvider dateAdapter={AdapterDateFns}>
      <Box p={3}>
        <Typography variant="h4" gutterBottom>
          Failed Claims Management
        </Typography>

        {/* Filters */}
        <Paper sx={{ p: 2, mb: 3 }}>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} sm={6} md={2}>
              <TextField
                fullWidth
                size="small"
                label="Facility ID"
                value={state.filters.facilityId}
                onChange={(e) => handleFilterChange('facilityId', e.target.value)}
              />
            </Grid>
            <Grid item xs={12} sm={6} md={2}>
              <FormControl fullWidth size="small">
                <InputLabel>Category</InputLabel>
                <Select
                  value={state.filters.failureCategory}
                  label="Category"
                  onChange={(e: SelectChangeEvent) =>
                    handleFilterChange('failureCategory', e.target.value)
                  }
                >
                  <MenuItem value="">All</MenuItem>
                  {failureCategories.map((category) => (
                    <MenuItem key={category} value={category}>
                      {category.replace('_', ' ').toUpperCase()}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={6} md={2}>
              <FormControl fullWidth size="small">
                <InputLabel>Status</InputLabel>
                <Select
                  value={state.filters.resolutionStatus}
                  label="Status"
                  onChange={(e: SelectChangeEvent) =>
                    handleFilterChange('resolutionStatus', e.target.value)
                  }
                >
                  <MenuItem value="">All</MenuItem>
                  {resolutionStatuses.map((status) => (
                    <MenuItem key={status} value={status}>
                      {status.toUpperCase()}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={6} md={2}>
              <DatePicker
                label="Start Date"
                value={state.filters.dateRange.startDate}
                onChange={(date) =>
                  handleFilterChange('dateRange', {
                    ...state.filters.dateRange,
                    startDate: date,
                  })
                }
                slotProps={{
                  textField: { size: 'small', fullWidth: true },
                }}
              />
            </Grid>
            <Grid item xs={12} sm={6} md={2}>
              <DatePicker
                label="End Date"
                value={state.filters.dateRange.endDate}
                onChange={(date) =>
                  handleFilterChange('dateRange', {
                    ...state.filters.dateRange,
                    endDate: date,
                  })
                }
                slotProps={{
                  textField: { size: 'small', fullWidth: true },
                }}
              />
            </Grid>
            <Grid item xs={12} sm={6} md={2}>
              <Button
                fullWidth
                variant="outlined"
                startIcon={<FilterIcon />}
                onClick={handleClearFilters}
                size="small"
              >
                Clear Filters
              </Button>
            </Grid>
          </Grid>
        </Paper>

        {/* Data Grid */}
        <Paper sx={{ height: 600, width: '100%' }}>
          {isLoading ? (
            <LoadingSpinner />
          ) : (
            <DataGrid
              rows={failedClaimsData?.failedClaims || []}
              columns={columns}
              checkboxSelection
              disableRowSelectionOnClick
              onRowSelectionModelChange={(newSelection) => {
                setState(prev => ({ ...prev, selectedClaims: newSelection }));
              }}
              rowSelectionModel={state.selectedClaims}
              slots={{
                toolbar: CustomToolbar,
              }}
              pageSizeOptions={[25, 50, 100]}
              initialState={{
                pagination: {
                  paginationModel: { page: 0, pageSize: 25 },
                },
              }}
              sx={{
                border: 0,
                '& .MuiDataGrid-cell:focus': {
                  outline: 'none',
                },
                '& .MuiDataGrid-row:hover': {
                  backgroundColor: 'rgba(0, 0, 0, 0.04)',
                },
              }}
            />
          )}
        </Paper>

        {/* Modals */}
        {state.detailModalOpen && state.selectedClaim && (
          <FailedClaimDetailModal
            claim={state.selectedClaim}
            open={state.detailModalOpen}
            onClose={() =>
              setState(prev => ({ ...prev, detailModalOpen: false, selectedClaim: null }))
            }
          />
        )}

        {state.resolutionModalOpen && state.selectedClaim && (
          <ClaimResolutionModal
            claim={state.selectedClaim}
            open={state.resolutionModalOpen}
            onClose={() =>
              setState(prev => ({ ...prev, resolutionModalOpen: false, selectedClaim: null }))
            }
            onResolve={(resolutionData) => {
              resolveMutation.mutate({
                claimId: state.selectedClaim!.id.toString(),
                ...resolutionData,
              });
            }}
            loading={resolveMutation.isLoading}
          />
        )}

        {/* Assignment Modal */}
        <Dialog
          open={state.assignModalOpen}
          onClose={() => setState(prev => ({ ...prev, assignModalOpen: false }))}
          maxWidth="sm"
          fullWidth
        >
          <DialogTitle>Assign Claims</DialogTitle>
          <DialogContent>
            <TextField
              fullWidth
              label="Assign To"
              margin="normal"
              helperText={`Assigning ${state.selectedClaims.length} claim(s)`}
            />
          </DialogContent>
          <DialogActions>
            <Button
              onClick={() => setState(prev => ({ ...prev, assignModalOpen: false }))}
            >
              Cancel
            </Button>
            <Button
              variant="contained"
              onClick={() => {
                // Handle assignment
                setState(prev => ({ ...prev, assignModalOpen: false }));
              }}
            >
              Assign
            </Button>
          </DialogActions>
        </Dialog>
      </Box>
    </LocalizationProvider>
  );
};

export default FailedClaims;